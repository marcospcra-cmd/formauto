"""
qwen_navegador.py — IA Qwen SEM API, usando a conta logada no navegador.
=========================================================================

COMO FUNCIONA (visão geral para o restante do sistema):
----------------------------------------------------------------------------
Este módulo NÃO usa nenhuma chave de API nem serviço local (Ollama etc.).
Ele automatiza o site oficial do Qwen (chat.qwen.ai) com Playwright e
reutiliza a SESSÃO LOGADA da conta do usuário:

1. PERFIL PERSISTENTE: usamos um "user data directory" dedicado
   (`~/.autodoc_qwen_profile`). No primeiro uso, uma janela visível abre no
   chat do Qwen e o usuário faz login normalmente (e-mail/senha/QR). O
   cookie/token do login fica gravado nesse perfil — nas próximas execuções
   o navegador abre JÁ autenticado, sem interação.

2. ENVIO DO PROMPT: a automação localiza a caixa de texto do chat por uma
   cascata de seletores (contenteditable / textarea / aria-labels em PT e
   EN), escreve o prompt e pressiona Enter.

3. ESPERA E CAPTURA: aguarda a resposta terminar observando o DOM — a
   conversa é considerada estável quando o número de mensagens não muda por
   N ciclos consecutivos e nenhum botão "stop/loading" está ativo; então
   captura o texto da ÚLTIMA mensagem do assistente.

4. ANTI-BLOQUEIO DE AUTOMAÇÃO: aplica patches leves via CDP (navigator.
   webdriver = false, remoção da flag "--enable-automation") para reduzir
   detecções simples de bot — o fluxo principal continua sendo o uso normal
   do site pela conta do usuário.

Todo o acesso ao navegador mora AQUI; `extractor.py` apenas delega para
esta camada quando o modo "navegador" está selecionado (padrão).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("qwen_navegador")

# ---------------------------------------------------------------------------
# Constantes de configuração do navegador
# ---------------------------------------------------------------------------
URL_CHAT: str = "https://chat.qwen.ai/"
PERFIL_DIR: Path = Path.home() / ".autodoc_qwen_profile"
USER_AGENT: str = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Seletores candidatos para a caixa de mensagem do chat (ordem = prioridade).
SELETORES_INPUT: tuple[str, ...] = (
    "div[contenteditable='true']",
    "textarea[aria-label*='message' i]",
    "textarea[placeholder*='messagem' i]",
    "textarea[placeholder*='message' i]",
    "textarea[placeholder*='Envie' i]",
    "textarea",
    "[role='textbox']",
)

# Seletores candidatos para as bolhas de conversa no feed.
SELETORES_MENSAGENS: tuple[str, ...] = (
    "[class*='message-content']",
    "[class*='markdown']",
    "[data-message-id]",
    "[class*='prose']",
)


class ErroQwenNavegador(RuntimeError):
    """Erro amigável da camada de navegador (login ausente, site fora etc.)."""


@dataclass
class ConfigQwenNavegador:
    """
    Parâmetros de execução do Qwen via navegador.

    Attributes:
        perfil: pasta do perfil Chromium persistente (mantém o login).
        url_chat: URL do chat do Qwen.
        timeout_envio: segundos máximos esperando a caixa de texto aparecer.
        timeout_resposta: segundos máximos esperando a resposta concluir.
        estavel_ciclos: quantos ciclos de 1s sem mudança de DOM indicam
            "resposta terminada".
        headless: True apenas em testes automatizados; o fluxo real usa
            janela visível (necessário para login e mais confiável).
    """
    perfil: Path = PERFIL_DIR
    url_chat: str = URL_CHAT
    timeout_envio: float = 25.0
    timeout_resposta: float = 180.0
    estavel_ciclos: int = 3
    headless: bool = False

    @classmethod
    def carregar(cls) -> "ConfigQwenNavegador":
        """Monta a config a partir de variáveis de ambiente (opcional)."""
        import os
        cfg = cls()
        if os.environ.get("AUTODOC_QWEN_URL", "").strip():
            cfg.url_chat = os.environ["AUTODOC_QWEN_URL"].strip()
        if os.environ.get("AUTODOC_QWEN_PROFILE", "").strip():
            cfg.perfil = Path(os.environ["AUTODOC_QWEN_PROFILE"]).expanduser()
        if os.environ.get("AUTODOC_QWEN_HEADLESS", "") == "1":
            cfg.headless = True
        return cfg


def _comandos_iniciais() -> list[str]:
    """Flags de launch que reduzem a pegada de automação do Chromium."""
    return [
        "--disable-blink-features=AutomationControlled",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
    ]


_PATCH_WEBDRIVER: str = """
// Remove signal clássico de bot: navigator.webdriver === true.
Object.defineProperty(Navigator.prototype, 'webdriver', {get: () => undefined});
"""


class MotorQwenNavegador:
    """
    Cliente de IA que dirige o navegador (Playwright) com a sessão logada.

    Uso típico::

        motor = MotorQwenNavegador(ConfigQwenNavegador.carregar())
        resposta = motor.perguntar("Extraia todos os pares chave-valor ...")

    A classe gerencia UMA página por vez dentro de um contexto persistente;
    nada de HTTP direto à API do Qwen — tudo passa pela interface web.
    """

    def __init__(self, cfg: Optional[ConfigQwenNavegador] = None) -> None:
        """Armazena a configuração; o navegador só abre sob demanda."""
        self.cfg: ConfigQwenNavegador = cfg or ConfigQwenNavegador.carregar()
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None

    # ------------------------------------------------------------ ciclo de vida
    def _garantir_navegador(self) -> Any:
        """
        Abre (ou reaproveita) o Chromium com o perfil persistente.

        Returns:
            Página (`playwright Page`) pronta para uso.

        Raises:
            ErroQwenNavegador: se o Playwright não estiver instalado.
        """
        if self._page is not None and not self._page.is_closed():
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - dependência declarada
            raise ErroQwenNavegador(
                "Playwright não instalado. Rode: pip install playwright "
                "&& playwright install chromium") from exc

        self.cfg.perfil.mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch_persistent_context(
                user_data_dir=str(self.cfg.perfil),
                headless=self.cfg.headless,
                args=_comandos_iniciais(),
                ignore_default_args=["--enable-automation"],
                user_agent=USER_AGENT,
                viewport={"width": 1380, "height": 900},
            )
        except Exception as exc:  # noqa: BLE001 - fronteira de robustez
            self._fechar()
            raise ErroQwenNavegador(f"Falha ao abrir o Chromium: {exc}") from exc

        # Patch anti-webdriver em TODAS as páginas do contexto.
        try:
            self._browser.add_init_script(_PATCH_WEBDRIVER)
        except Exception:  # noqa: BLE001 - patch é melhor esforço
            logger.debug("Patch init-script indisponível (ignorado).")

        pages = self._browser.pages
        self._page = pages[0] if pages else self._browser.new_page()
        return self._page

    def _fechar(self) -> None:
        """Encerra navegador + driver com segurança idempotente."""
        for closer in (
            lambda: self._browser and self._browser.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._browser = self._page = self._pw = None

    # -------------------------------------------------------------- navegação
    def abrir_chat(self, novo_chat: bool = True) -> None:
        """
        Vai até a página do chat do Qwen (e tenta iniciar uma conversa nova).

        Args:
            novo_chat: se True, clica no botão "+ New chat" quando existir —
                assim cada análise parte de um contexto limpo.
        """
        page = self._garantir_navegador()
        try:
            page.goto(self.cfg.url_chat, wait_until="domcontentloaded",
                      timeout=45_000)
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception as exc:  # noqa: BLE001 - timeouts de rede são comuns
            raise ErroQwenNavegador(
                f"Não consegui abrir {self.cfg.url_chat} ({type(exc).__name__}). "
                "Verifique sua conexão.") from exc
        if novo_chat:
            for sel in ("button[aria-label*='new chat' i]",
                        "a[aria-label*='new' i]"):
                try:
                    el = page.query_selector(sel)
                    if el:
                        el.click(timeout=1500)
                        break
                except Exception:  # noqa: BLE001
                    continue

    def esta_logado(self) -> bool:
        """
        Heurística de sessão: a caixa de mensagem só existe para usuários
        autenticados; páginas de login/redirecionamento não a exibem.
        """
        page = self._garantir_navegador()
        if any(k in (page.url or "").lower()
               for k in ("sign-in", "login", "auth")):
            return False
        return self._achar_input(timeout_ms=2500) is not None

    # ------------------------------------------------------------------- envio
    def _achar_input(self, timeout_ms: float = 800) -> Optional[Any]:
        """Varre os seletores candidatos e devolve o primeiro campo visível."""
        page = self._page
        if page is None:
            return None
        for sel in SELETORES_INPUT:
            try:
                el = page.wait_for_selector(sel, state="visible",
                                            timeout=min(timeout_ms, 800))
                if el:
                    return el
            except Exception:  # noqa: BLE001 - próximo seletor
                continue
        return None

    def _contar_mensagens(self) -> int:
        """Total de bolhas de conversa presentes no feed (baseline p/ espera)."""
        page = self._page
        assert page is not None
        total = 0
        for sel in SELETORES_MENSAGENS:
            try:
                total += len(page.query_selector_all(sel))
            except Exception:  # noqa: BLE001
                continue
        return total

    def _gerando(self) -> bool:
        """True se houver indicador 'stop/gerando' ativo na página."""
        page = self._page
        assert page is not None
        for sel in ("button[aria-label*='stop' i]",
                    "button[data-testid*='stop']"):
            try:
                if page.query_selector(sel):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def enviar_prompt(self, prompt: str) -> None:
        """
        Escreve o prompt na caixa do chat e dispara o envio (Enter).

        Raises:
            ErroQwenNavegador: campo não encontrado (possivelmente deslogado).
        """
        page = self._garantir_navegador()
        campo = self._achar_input(timeout_ms=self.cfg.timeout_envio * 1000)
        if campo is None:
            raise ErroQwenNavegador(
                "Não encontrei a caixa de mensagem do Qwen. Se a janela mostrou "
                "a tela de login, faça login na janela que abri e tente novamente.")
        try:
            campo.click()
            # Digitação em pedaços evita colar texto gigante de uma vez
            # (alguns chats truncam paste muito grande).
            page.keyboard.type(prompt[:4000], delay=2) if len(prompt) <= 4000 \
                else campo.fill(prompt[:4000])
            page.keyboard.press("Enter")
        except Exception as exc:  # noqa: BLE001
            raise ErroQwenNavegador(f"Falha ao enviar o prompt: {exc}") from exc

    # ------------------------------------------------------------- captura
    def aguardar_resposta(self) -> str:
        """
        Espera a geração terminar e devolve o texto da última resposta.

        Estratégia: polling de 1s; a resposta é considerada concluída quando
        (a) o número de mensagens aumentou em relação ao baseline, (b) não há
        botão "stop" ativo e (c) o DOM fica estável por `estavel_ciclos`.

        Raises:
            ErroQwenNavegador: tempo esgotado ou resposta vazia.
        """
        page = self._page
        assert page is not None
        baseline = self._contar_mensagens()
        limite = time.monotonic() + self.cfg.timeout_resposta
        ultimo_total, estaveis = baseline, 0
        while time.monotonic() < limite:
            time.sleep(1.0)
            total = self._contar_mensagens()
            if total > baseline and not self._gerando():
                estaveis = estaveis + 1 if total == ultimo_total else 0
            ultimo_total = max(total, ultimo_total)
            if estaveis >= self.cfg.estavel_ciclos:
                break
        else:
            raise ErroQwenNavegador(
                "Tempo esgotado aguardando a resposta do Qwen no navegador.")

        texto = self._ultima_resposta()
        if not texto.strip():
            raise ErroQwenNavegador("O Qwen respondeu, mas não capturei o texto.")
        return texto

    def _ultima_resposta(self) -> str:
        """Captura o textContent da última bolha de assistente disponível."""
        page = self._page
        assert page is not None
        candidatos: list[str] = []
        for sel in SELETORES_MENSAGENS:
            try:
                els = page.query_selector_all(sel)
            except Exception:  # noqa: BLE001
                continue
            for el in els:
                try:
                    t = (el.text_content() or "").strip()
                    if t:
                        candidatos.append(t)
                except Exception:  # noqa: BLE001
                    continue
        if not candidatos:
            # fallback bruto: último bloco <p>/<div> grande do feed
            try:
                blocks = page.evaluate("""() => Array.from(
                    document.querySelectorAll('p, div'))
                    .map(e => e.innerText || '').filter(t => t.length > 40)""")
                candidatos = [str(b) for b in blocks] if blocks else []
            except Exception:  # noqa: BLE001
                candidatos = []
        return candidatos[-1] if candidatos else ""

    # ------------------------------------------------------------ API pública
    def perguntar(self, prompt: str) -> str:
        """
        Ciclo completo: abre o chat, envia o prompt e devolve a resposta crua.

        Esta é a única função que o resto do sistema precisa conhecer.
        """
        try:
            self.abrir_chat(novo_chat=True)
            self.enviar_prompt(prompt)
            return self.aguardar_resposta()
        finally:
            self._fechar()


def disponivel() -> bool:
    """
    Indica rapidamente se o modo navegador está pronto para uso.

    Considera disponível quando: Playwright instalado + Chromium presente +
    perfil com histórico de login (arquivos de cookies do Chromium).
    Não abre janela — checagem barata para a GUI decidir o rótulo do motor.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return False
    tem_perfil = any(PERFIL_DIR.glob("Default/Cookies")) or any(
        PERFIL_DIR.glob("**/Cookies"))
    return tem_perfil


def testar_conexao(mostrar_janela: bool = True,
                   instrucao_cb: Optional[Any] = None) -> tuple[bool, str]:
    """
    Teste de conectividade/sessão usado pelo diálogo de configuração da GUI.

    Abre o chat do Qwen no navegador persistente; se o usuário ainda não fez
    login, a janela permanece aberta para que ele autentique (o perfil grava
    a sessão automaticamente).

    Args:
        mostrar_janela: sempre True no fluxo humano (headless não permite
            login).
        instrucao_cb: callback opcional(str) para ecoar instruções à UI.

    Returns:
        (logado, mensagem humana explicando o estado).
    """
    cfg = ConfigQwenNavegador.carregar()
    cfg.headless = False  # login exige janela visível
    motor = MotorQwenNavegador(cfg)
    try:
        motor.abrir_chat(novo_chat=False)
        if motor.esta_logado():
            return True, ("✔ Sessão Qwen encontrada! O programa já pode usar sua "
                          "conta logada no navegador para analisar os documentos.")
        if callable(instrucao_cb):
            instrucao_cb("Faça LOGIN do Qwen na janela do navegador que acabei de "
                         "abrir. Quando aparecer a tela do chat, feche esta janela "
                         "de teste e clique em 'Salvar e testar' novamente.")
        return False, ("⚠ Nenhum login do Qwen encontrado neste perfil.\n"
                       "Abri a janela do chat: entre com sua conta (e-mail, QR ou "
                       "Google). O login fica salvo e será reaproveitado nas "
                       "próximas análises — sem API, sem Ollama.")
    except ErroQwenNavegador as exc:
        return False, f"✖ {exc}"
    except Exception as exc:  # noqa: BLE001 - fronteira de robustez
        return False, f"✖ Falha inesperada no navegador: {exc}"
    # NOTA: não fechamos o browser aqui de propósito quando deslogado,
    # para deixar a janela aberta durante o login do usuário.


def _limpar_cerca(resposta: str) -> str:
    """Remove cercas ```json ... ``` que modelos adoram adicionar."""
    m = re.search(r"```(?:json)?\s*(.*?)```", resposta, flags=re.DOTALL)
    return m.group(1) if m else resposta


def extrair_json_da_resposta(resposta: str) -> list[Any]:
    """
    Converte a resposta livre do chat em lista de dicts tolerante.

    Aceita JSON puro, JSON dentro de cercas de código e até objetos soltos.
    """
    bruto = _limpar_cerca(resposta)
    m = re.search(r"\[.*\]", bruto, flags=re.DOTALL)
    if m:
        try:
            dados = json.loads(m.group(0))
            if isinstance(dados, list):
                return dados
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{.*\}", bruto, flags=re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            return [obj] if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Nenhum JSON reconhecível na resposta do Qwen:\n{resposta[:400]}")


if __name__ == "__main__":  # teste manual: python qwen_navegador.py "olá"
    import sys
    logging.basicConfig(level=logging.INFO)
    alvo = " ".join(sys.argv[1:]) or "Responda apenas: OK"
    print(MotorQwenNavegador().perguntar(alvo)[:2000])
