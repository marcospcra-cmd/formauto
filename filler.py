"""
filler.py — Mapeamento semântico e execução de preenchimento (Web/PDF/TXT).
================================================================================

COMO FUNCIONA O MAPEAMENTO SEMÂNTICO:
-------------------------------------
Correspondência exata de strings falha em mundos reais ("Nome Completo" vs
"nome_completo" vs "Full Name"). Este módulo compara o RÓTULO de cada campo do
destino com as CHAVES extraídas da origem usando similaridade de texto:

1. Backend preferido: `sentence-transformers` (modelo leve all-MiniLM-L6-v2),
   que produz embeddings e mede similaridade de cosseno — captura sinônimos
   conceituais ("Endereço" ~ "Localização").
2. Fallback automático: `thefuzz` (fuzzywuzzy) com `token_sort_ratio` +
   normalização (caixa baixa, remoção de acentos/pontuação/underscores).
   Ativado se sentence-transformers não estiver instalado OU se o download do
   modelo falhar (modo offline).

Regra de decisão (limiar configurável, padrão 0.75):
    score >= LIMIAR  -> mapeamento AUTOMÁTICO (status OK)
    score <  LIMIAR  -> status REVISAO_NECESSARIA (aparece na GUI para o
                        usuário escolher manualmente via Combobox)

O resultado é uma lista de `MapeamentoCampo`, compartilhada entre os executores
(Web/PDF/TXT) e a interface gráfica.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from extractor import DadoExtraido, Fonte

logger = logging.getLogger("filler")

LIMIAR_SIMILARIDADE: float = 0.75  # >= isso => mapeamento automático


class StatusCampo(str, Enum):
    """Estado de um campo de destino após o mapeamento."""
    OK = "ok"                       # confiança alta, pronto p/ preencher
    REVISAO = "revisao_necessaria"  # similaridade abaixo do limiar
    SEM_DADO = "sem_correspondencia"
    PREENCHIDO = "preenchido"
    FALHA = "falha"


@dataclass
class CampoDestino:
    """
    Representa um alvo de preenchimento independentemente da tecnologia.

    Attributes:
        rotulo: Texto humano do campo (label, placeholder, nome de widget...).
        identificador: `name`/`id` (web), `/FieldName` (AcroForm) ou offset (txt).
        tipo: input|select|textarea|pdf_field|pdf_visual|txt_key.
        coordenadas: (x0, top, x1, bottom) para PDFs visuais sem AcroForm.
        opcoes: valores válidos para selects.
    """
    rotulo: str
    identificador: str
    tipo: str = "input"
    coordenadas: Optional[tuple[float, float, float, float]] = None
    opcoes: list[str] = field(default_factory=list)


@dataclass
class MapeamentoCampo:
    """
    Par (campo de destino -> dado de origem) com score de confiança.

    Attributes:
        campo: O widget/lacuna do formulário alvo.
        dado: O `DadoExtraido` escolhido (heurístico ou pela IA Qwen).
        score: Confiança final do encaixe (0..1).
        status: Estado atual (OK / REVISAO / SEM_DADO / PREENCHIDO / FALHA).
        valor_final: O que será efetivamente gravado no destino.
        justificativa: Explicação legível de POR QUE este dado foi encaixado
            neste campo — preenchida pela IA Qwen (ou pelo matching heurístico)
            e exibida na GUI para permitir revisão informada pelo usuário.
        origem_encaixe: Quem decidiu o encaixe: "qwen" (IA real), "heuristica"
            (similaridade fuzzy/neural) ou "manual" (override do usuário).
    """
    campo: CampoDestino
    dado: Optional[DadoExtraido] = None
    score: float = 0.0
    status: StatusCampo = StatusCampo.SEM_DADO
    valor_final: str = ""
    justificativa: str = ""
    origem_encaixe: str = "heuristica"

    @property
    def precisa_revisao(self) -> bool:
        """True quando o usuário deve validar/escolher manualmente na GUI."""
        return self.status in (StatusCampo.REVISAO, StatusCampo.SEM_DADO)


# ---------------------------------------------------------------------------
# Normalização linguística usada por ambos os backends
# ---------------------------------------------------------------------------
def _normalizar(texto: str) -> str:
    """minúsculas + remove acentos + remove pontuação/separadores técnicos."""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[_\-:.]", " ", t)
    t = re.sub(r"[^a-z0-9ñ\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ===========================================================================
# Motor de similaridade (sentence-transformers com fallback thefuzz)
# ===========================================================================
class Similarizador:
    """
    Calcula similaridade 0..1 entre rótulos. Carrega o modelo neural apenas
    uma vez (lazy) e degrada graciosamente para fuzzy matching.
    """

    def __init__(self, usar_neural: bool = True) -> None:
        """
        Args:
            usar_neural: desabilite para forçar o backend fuzzy (testes/offline).
        """
        self.usar_neural = usar_neural
        self._modelo: Any = None
        self._tentou_carregar: bool = False
        self.backend: str = "pending"

    def _garantir_modelo(self) -> bool:
        """Retorna True se o modelo neural está disponível; tenta carregar 1x."""
        if self._tentou_carregar:
            return self._modelo is not None
        self._tentou_carregar = True
        if not self.usar_neural:
            self.backend = "thefuzz"
            return False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Modelo leve (~80MB). Em ambiente offline o ImportError/OSError
            # cai no except e ativamos o fallback fuzzy automaticamente.
            self._modelo = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "sentence-transformers"
            logger.info("Backend semântico: sentence-transformers (all-MiniLM-L6-v2)")
            return True
        except Exception as exc:  # noqa: BLE001 - ausência de modelo/rede é esperada
            logger.warning("sentence-transformers indisponível (%s); usando thefuzz.", exc)
            self.backend = "thefuzz"
            return False

    def similaridade(self, a: str, b: str) -> float:
        """Similaridade semântica entre duas strings, sempre no intervalo [0,1]."""
        na, nb = _normalizar(a), _normalizar(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0
        # Guard anti-falso-positivo do fuzzy: tokens de tamanho muito
        # desproporcionais ("uf" vs "nome completo") nunca devem casar.
        wa, wb = set(na.split()), set(nb.split())
        menor, maior = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        if menor and max(len(t) for t in menor) < 4 and not (menor & maior):
            return 0.0
        if self._garantir_modelo():
            try:
                emb = self._modelo.encode([na, nb])
                cos = float(emb @ emb.T / (
                    (float((emb[0] ** 2).sum()) ** 0.5) * float((emb[1] ** 2).sum()) ** 0.5
                )) if hasattr(emb, "__matmul__") else self._cosseno_list(emb)
                return max(0.0, min(1.0, cos))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha no cálculo neural (%s); fallback fuzzy.", exc)
        return self._fuzzy(na, nb)

    @staticmethod
    def _cosseno_list(emb: Any) -> float:
        """Cosseno manual para backends que retornam list[list[float]]."""
        import math
        va, vb = list(emb[0]), list(emb[1])
        dot = sum(x * y for x, y in zip(va, vb))
        n = math.sqrt(sum(x * x for x in va)) * math.sqrt(sum(y * y for y in vb))
        return dot / n if n else 0.0

    @staticmethod
    def _fuzzy(a: str, b: str) -> float:
        """
        Fallback fuzzy: combina token_sort_ratio (ordem-independente) com
        partial_ratio (substrings úteis p/ 'Nome' vs 'Nome Completo').
        """
        try:
            from thefuzz import fuzz  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instale 'thefuzz': pip install thefuzz python-Levenshtein") from exc
        s1 = fuzz.token_sort_ratio(a, b) / 100.0
        s2 = fuzz.partial_ratio(a, b) / 100.0
        s3 = fuzz.WRatio(a, b) / 100.0
        return max(s1, 0.6 * s2 + 0.4 * s3)


# ===========================================================================
# Núcleo de mapeamento
# ===========================================================================
class MapeadorSemanticos:
    """Cruza dados extraídos (origem) com campos de destino via similaridade."""

    def __init__(self, similarizador: Optional[Similarizador] = None,
                 limiar: float = LIMIAR_SIMILARIDADE) -> None:
        """Args: similarizador injetável (testes) e limiar de auto-mapeamento."""
        self.sim = similarizador or Similarizador()
        self.limiar = limiar

    def mapear(self, campos: list[CampoDestino],
               dados: list[DadoExtraido],
               encaixes_ia: Optional[list[Any]] = None) -> list[MapeamentoCampo]:
        """
        Estratégia de encaixe em DUAS CAMADAS:

        1. IA QWEN via navegador do usuário, sem API (camada preferencial): cada
           `EncaixeQwen` retornado por
           `consultar_qwen_encaixe()` já traz o par
           (rotulo_destino -> valor adaptado + justificativa). O rótulo é
           casado com os campos do destino (exact-after-normalization e, se
           necessário, fuzzy >= limiar). Quando há match, o campo recebe o
           encaixe da IA com confiança/justificativa dela — sem passar pela
           heurística. Os dados de origem cuja chave foi nomeada pela própria
           IA (`Fonte.IA_QWEN`) são anexados ao mapa para rastreabilidade.

        2. HEURÍSTICA SEMÂNTICA (camada de fallback): para os campos que a IA
           NÃO cobriu (IA ausente, falhou ou omitiu o campo por não haver dado
           no documento), roda o matching clássico de similaridade entre o
           rótulo do campo e as chaves extraídas — score >= limiar => OK,
           abaixo => REVISAO_NECESÁRIA, sem candidato => SEM_DADO.

        Args:
            campos: Campos coletados do destino (web/PDF/TXT).
            dados: Todos os `DadoExtraido` da origem.
            encaixes_ia: Lista de `extractor.EncaixeQwen` (ou None) produzida
                durante a extração; tipada como Any aqui para evitar ciclo de
                import em tempo de execução.

        Returns:
            Lista de `MapeamentoCampo`; cada item carrega `origem_encaixe`
            ("qwen" | "heuristica") e `justificativa` exibíveis na GUI.
        """
        mapas: list[MapeamentoCampo] = []
        cobertos: set[int] = set()

        # ---------- Camada 1: decisões de encaixe vindas do Qwen ----------
        por_chave = {d.chave_original: d for d in dados}
        for e in (encaixes_ia or []):
            alvo = self._casar_rotulo(getattr(e, "rotulo_destino", ""), campos)
            if alvo is None or alvo in cobertos:
                continue  # IA citou um rótulo que não existe no formulário
            cobertos.add(alvo)
            campo = campos[alvo]
            # tenta vincular ao dado de origem correspondente (rastreabilidade)
            d = por_chave.get(getattr(e, "chave_origem", "")) \
                or por_chave.get(campo.rotulo) \
                or next((x for x in dados
                         if x.fonte == Fonte.IA_QWEN.value
                         and _normalizar(x.chave_original) == _normalizar(campo.rotulo)),
                        None)
            conf = float(getattr(e, "confianca", 0.8))
            just = str(getattr(e, "justificativa", "") or "").strip()
            mapas.append(MapeamentoCampo(
                campo=campo, dado=d, score=conf,
                status=StatusCampo.OK if conf >= self.limiar else StatusCampo.REVISAO,
                valor_final=str(getattr(e, "valor", "")),
                justificativa=just or f"encaixou o dado '{getattr(e, 'chave_origem', '?')}' neste campo",
                origem_encaixe="qwen"))

        # ---------- Camada 2: matching heurístico p/ campos restantes ----------
        chaves = [d.chave_original for d in dados]
        for i_c, campo in enumerate(campos):
            if i_c in cobertos:
                continue
            melhor_i, melhor_s = -1, 0.0
            for i, chave in enumerate(chaves):
                s = self.sim.similaridade(campo.rotulo, chave)
                # bônus modesto quando o `identificador` técnico contém a chave
                if campo.identificador and _normalizar(chave) in _normalizar(campo.identificador):
                    s = min(1.0, s + 0.10)
                if s > melhor_s:
                    melhor_i, melhor_s = i, s
            if melhor_i >= 0 and melhor_s >= self.limiar:
                d = dados[melhor_i]
                mapas.append(MapeamentoCampo(
                    campo, d, melhor_s, StatusCampo.OK, d.valor,
                    justificativa=(f"similaridade {melhor_s:.0%} entre o rótulo "
                                  f"'{campo.rotulo}' e a chave '{d.chave_original}'"),
                    origem_encaixe="heuristica"))
            elif melhor_i >= 0:
                d = dados[melhor_i]
                mapas.append(MapeamentoCampo(
                    campo, d, melhor_s, StatusCampo.REVISAO, d.valor,
                    justificativa=(f"melhor candidato '{d.chave_original}' atingiu apenas "
                                   f"{melhor_s:.0%} (< {self.limiar:.0%}) — revisar"),
                    origem_encaixe="heuristica"))
            else:
                mapas.append(MapeamentoCampo(
                    campo, None, 0.0, StatusCampo.SEM_DADO, "",
                    justificativa="nenhum dado da origem parece corresponder a este campo",
                    origem_encaixe="heuristica"))
        return mapas

    def _casar_rotulo(self, rotulo_ia: str, campos: list[CampoDestino]) -> Optional[int]:
        """
        Localiza o índice do campo do formulário cujo rótulo corresponde ao
        `rotulo_destino` devolvido pelo Qwen. Tenta igualdade normalizada e,
        em seguida, fuzzy acima do limiar (a IA costuma citar o rótulo exato,
        mas pode parafrasear).
        """
        if not rotulo_ia:
            return None
        alvo = _normalizar(rotulo_ia)
        for i, c in enumerate(campos):
            if _normalizar(c.rotulo) == alvo:
                return i
        melhor_i, melhor_s = None, 0.0
        for i, c in enumerate(campos):
            s = self.sim.similaridade(rotulo_ia, c.rotulo)
            if s > melhor_s:
                melhor_i, melhor_s = i, s
        return melhor_i if melhor_s >= self.limiar else None

    def melhor_candidatos(self, campo: CampoDestino, dados: list[DadoExtraido],
                          n: int = 5) -> list[tuple[DadoExtraido, float]]:
        """Top-N candidatos para popular o dropdown de revisão manual na GUI."""
        pares = [(d, self.sim.similaridade(campo.rotulo, d.chave_original)) for d in dados]
        pares.sort(key=lambda x: x[1], reverse=True)
        return pares[:n]


# ===========================================================================
# Destinos: inspetores (coletam campos) + executores (preenchem)
# ===========================================================================
class PreenchedorWeb:
    """
    Inspeta e preenche formulários HTML com Playwright.

    Coleta inputs/selects/textareas e usa 5 fontes de rótulo por elemento:
    name, id, placeholder, aria-label e o texto do <label> mais próximo
    (for=[id] ou ancestral) — todas entram no cálculo semântico.
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 15000) -> None:
        """Args: headless do browser e timeout de navegação (site fora do ar)."""
        self.headless = headless
        self.timeout_ms = timeout_ms

    # -- inspeção ------------------------------------------------------------
    async def coletar_campos_async(self, url: str) -> tuple[list[CampoDestino], Any, Any]:
        """Abre a página e devolve (campos, page, browser) para reuso no fill."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instale playwright: pip install playwright && playwright install chromium") from exc

        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(headless=self.headless)
        except Exception as exc:  # navegador não instalado
            await pw.stop()
            raise RuntimeError(f"Chromium indisponível ({exc}). Rode 'playwright install chromium'.") from exc
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            await browser.close()
            await pw.stop()
            raise ConnectionError(f"Site inacessível ({url}): {exc}") from exc

        raw = await page.evaluate(_JS_COLETAR_CAMPOS)
        campos = [
            CampoDestino(
                rotulo=r.get("ariaLabel") or r.get("label") or r.get("placeholder")
                       or r.get("name") or r.get("id") or f"campo_{i}",
                identificador=r.get("name") or r.get("id") or f"__idx{i}",
                tipo=r["tag"],
                opcoes=r.get("options", []),
            )
            for i, r in enumerate(raw)
        ]
        return campos, page, (browser, pw)

    def coletar_campos(self, url: str) -> list[CampoDestino]:
        """Versão síncrona (abre e fecha o browser internamente)."""
        import asyncio

        async def _run() -> list[CampoDestino]:
            campos, _, ctx = await self.coletar_campos_async(url)
            browser, pw = ctx
            await browser.close()
            await pw.stop()
            return campos
        return asyncio.run(_run())

    # -- preenchimento ---------------------------------------------------------
    async def preencher_async(self, url: str, mapas: list[MapeamentoCampo],
                              dry_run: bool = False) -> list[MapeamentoCampo]:
        """
        Executa o preenchimento. Em `dry_run`, valida se cada alvo é localizável
        e coloriza, mas NÃO envia/submete nada. Após sucesso, marca o fundo de
        verde; em falha, laranja — exatamente como especificado.
        """
        campos, page, ctx = await self.coletar_campos_async(url)
        browser, pw = ctx
        mapa_por_id = {m.campo.identificador: m for m in mapas}
        try:
            for m in mapas:
                sel = _seletor_para(m.campo)
                try:
                    el = page.locator(sel).first
                    await el.wait_for(state="attached", timeout=3000)
                    if dry_run:
                        await _marcar(page, sel, cor="#c8f7c5" if not m.precisa_revisao else "#ffd8a8")
                        m.status = StatusCampo.OK if not m.precisa_revisao else m.status
                        continue
                    if m.precisa_revisao and not m.valor_final:
                        m.status = StatusCampo.REVISAO
                        await _marcar(page, sel, cor="#ffd8a8")
                        continue
                    if m.campo.tipo == "select":
                        await el.select_option(label=m.valor_final)
                    else:
                        await el.fill(m.valor_final)
                    await _marcar(page, sel, cor="#c8f7c5")  # verde: sucesso
                    m.status = StatusCampo.PREENCHIDO
                except Exception as exc:  # noqa: BLE001 - campo individual pode falhar
                    logger.warning("Falha ao preencher %s: %s", m.campo.identificador, exc)
                    try:
                        await _marcar(page, sel, cor="#ffb347")  # laranja: falha
                    except Exception:  # noqa: BLE001
                        pass
                    m.status = StatusCampo.FALHA
        finally:
            if not dry_run:
                await page.screenshot(path="resultado_web.png")
            await browser.close()
            await pw.stop()
        _ = mapa_por_id  # mantido para depuração futura
        return mapas

    def preencher(self, url: str, mapas: list[MapeamentoCampo],
                  dry_run: bool = False) -> list[MapeamentoCampo]:
        """API síncrona para a GUI (Tkinter não roda dentro de um event loop)."""
        import asyncio
        return asyncio.run(self.preencher_async(url, mapas, dry_run))


# JavaScript injetado: varre todos os controles de formulário e resolve o
# <label> associado (for=, wrap, aria-labelledby) para dar 5 sinais de rótulo.
_JS_COLETAR_CAMPOS = """
() => {
  const out = [];
  const els = document.querySelectorAll('input, select, textarea');
  els.forEach((el) => {
    if (el.type === 'hidden' || el.disabled) return;
    let label = '';
    if (el.labels && el.labels.length) label = el.labels[0].textContent.trim();
    if (!label && el.getAttribute('aria-labelledby')) {
      const ref = document.getElementById(el.getAttribute('aria-labelledby'));
      if (ref) label = ref.textContent.trim();
    }
    if (!label) {
      const guess = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (guess) label = guess.textContent.trim();
    }
    out.push({
      tag: el.tagName.toLowerCase() === 'select' ? 'select' : el.tagName.toLowerCase(),
      name: el.name || '',
      id: el.id || '',
      placeholder: el.placeholder || '',
      ariaLabel: el.getAttribute('aria-label') || '',
      label: label,
      options: el.tagName === 'SELECT'
        ? Array.from(el.options).map(o => o.textContent.trim()).filter(Boolean)
        : [],
    });
  });
  return out;
}
"""


def _seletor_para(campo: CampoDestino) -> str:
    """Constrói um seletor CSS estável a partir do identificador do campo."""
    if campo.identificador.startswith("__idx"):
        i = campo.identificador[5:]
        return f"input:nth-of-type(1), [data-idx='{i}']"
    safe = campo.identificador.replace("'", "\\'")
    return f"[name='{safe}'], #{safe}"


async def _marcar(page: Any, seletor: str, cor: str) -> None:
    """Aplica background-color inline para feedback visual (verde/laranja)."""
    await page.evaluate(
        """([sel, cor]) => {
            const el = document.querySelector(sel.split(', ')[0]);
            if (el) el.style.backgroundColor = cor;
        }""",
        [seletor, cor],
    )


# ---------------------------------------------------------------------------
# PDF: AcroForm nativo + escrita visual sobre lacunas
# ---------------------------------------------------------------------------
class PreenchedorPDF:
    """
    Suporta dois modos:
      * ACROFORM: campos nativos (`/Tx`, `/Btn`, `/Ch`) preenchidos via pypdf.
      * VISUAL: PDF sem campos interativos — `reportlab` desenha o texto por
        cima das coordenadas informadas nos `CampoDestino.coordenadas`.
    """

    def __init__(self, caminho_pdf: str | Path) -> None:
        """Args: caminho do PDF de destino (template)."""
        self.caminho = Path(caminho_pdf)
        if not self.caminho.exists():
            raise FileNotFoundError(f"PDF de destino não encontrado: {self.caminho}")

    def coletar_campos(self) -> list[CampoDestino]:
        """Lista os campos AcroForm existentes (rota vazia se PDF puramente visual)."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(self.caminho))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise PermissionError("PDF de destino protegido por senha.") from exc
            campos: list[CampoDestino] = []
            for name, f in (reader.get_fields() or {}).items():
                ft = str(f.get("/FT", "/Tx"))
                tipo = {"/Tx": "input", "/Btn": "select", "/Ch": "select"}.get(ft, "input")
                opts = [str(o) for o in (f.get("/_States_") or [])]
                campos.append(CampoDestino(rotulo=name, identificador=name,
                                           tipo=tipo, opcoes=opts))
            return campos
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Falha ao inspecionar PDF ({type(exc).__name__}): {exc}") from exc

    def preencher(self, mapas: list[MapeamentoCampo], saida: str | Path,
                  dry_run: bool = False) -> Path:
        """
        Aplica os mapeamentos e grava o PDF resultante.

        Estratégia híbrida automática: campos com `coordenadas` são escritos
        visualmente (overlay reportlab); demais seguem o fluxo AcroForm (pypdf).
        """
        saida = Path(saida)
        visuais = [m for m in mapas if m.campo.coordenadas and m.valor_final]
        acro = [m for m in mapas if not (m.campo.coordenadas and m.valor_final)]

        base = self.caminho
        if acro:
            base = self._preencher_acroform(acro, saida.with_suffix(".acro.tmp.pdf"), dry_run)
        if visuais:
            self._overlay_visual(visuais, base, saida)
        else:
            if base != saida:
                saida.write_bytes(Path(base).read_bytes())
            Path(base).unlink(missing_ok=True)

        for m in mapas:
            if m.precisa_revisao and not m.valor_final:
                m.status = StatusCampo.REVISAO
            elif m.status not in (StatusCampo.REVISAO, StatusCampo.SEM_DADO):
                m.status = StatusCampo.PREENCHIDO if not dry_run else StatusCampo.OK
        return saida

    def _preencher_acroform(self, mapas: list[MapeamentoCampo],
                            saida: Path, dry_run: bool) -> Path:
        """Escreve valores nos campos nativos via pypdf (auto_regenerate_fields)."""
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(str(self.caminho))
        writer = PdfWriter(clone_from=reader)
        # update_page_form_field_values espera chaves SEM a barra inicial
        # (ex.: "nome_completo", não "/nome_completo") — normalizamos aqui.
        valores = {m.campo.identificador.lstrip("/"): m.valor_final
                   for m in mapas if m.valor_final}
        for pagina in writer.pages:  # cobre todos os paginos com widgets
            try:
                writer.update_page_form_field_values(pagina, valores,
                                                     auto_regenerate=False)
            except Exception as exc:  # nem todo template expõe NeedAppearances
                logger.warning("update_page_form_field_values: %s", exc)
        if not dry_run:
            with open(saida, "wb") as fh:
                writer.write(fh)
        else:
            saida.write_bytes(self.caminho.read_bytes())  # dry-run: cópia idêntica
        return saida

    def _overlay_visual(self, mapas: list[MapeamentoCampo],
                        base: Path, saida: Path) -> None:
        """Desenha texto sobre as lacunas com reportlab (PDF 'achatado'/visual)."""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from pypdf import PdfReader, PdfWriter
        import io

        largura, altura = letter
        try:
            r = PdfReader(str(base))
            mb = r.mediabox
            largura, altura = float(mb.width), float(mb.height)
        except Exception:  # noqa: BLE001
            pass

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(largura, altura))
        for m in mapas:
            x0, top, _x1, _bot = m.campo.coordenadas  # pdfplumber usa origem no TOPO
            y = altura - top - 2  # converte para origem inferior do reportlab
            c.setFont("Helvetica", 9)
            c.drawString(x0, y, m.valor_final)
        c.save()
        buf.seek(0)

        overlay_reader = PdfReader(buf)
        base_reader = PdfReader(str(base))
        writer = PdfWriter(clone_from=base_reader)
        for i, pagina in enumerate(writer.pages):
            if i < len(overlay_reader.pages):
                pagina.merge_page(overlay_reader.pages[i])
        with open(saida, "wb") as fh:
            writer.write(fh)


# ---------------------------------------------------------------------------
# TXT: substitui/insere linhas "chave: valor"
# ---------------------------------------------------------------------------
class PreenchedorTXT:
    """Preenche templates .txt procurando a linha cujo rótulo casa com o campo."""

    def __init__(self, caminho_txt: str | Path) -> None:
        """Args: caminho do arquivo de destino (será lido como template)."""
        self.caminho = Path(caminho_txt)
        if not self.caminho.exists():
            raise FileNotFoundError(f"TXT de destino não encontrado: {self.caminho}")

    def coletar_campos(self) -> list[CampoDestino]:
        """Cada linha 'Rótulo:' ou 'Rótulo = ___' vira um campo de destino."""
        campos: list[CampoDestino] = []
        texto = self.caminho.read_text(encoding="utf-8", errors="replace")
        for i, linha in enumerate(texto.splitlines()):
            m = re.match(r"^\s*([^:=\n]{2,60}?)\s*[:=]\s*([_\s]*|\{\{.*?\}\})$", linha)
            if m:
                campos.append(CampoDestino(rotulo=m.group(1).strip(),
                                           identificador=f"linha{i}:{m.group(1).strip()}",
                                           tipo="txt_key"))
        return campos

    def preencher(self, mapas: list[MapeamentoCampo], saida: str | Path,
                  dry_run: bool = False) -> Path:
        """Substitui as linhas de destino por 'Rótulo: valor_final'."""
        saida = Path(saida)
        texto = self.caminho.read_text(encoding="utf-8", errors="replace")
        linhas = texto.splitlines()
        por_linha = {int(m.campo.identificador.split(":")[0].replace("linha", "")): m
                     for m in mapas if m.campo.identificador.startswith("linha")}
        for idx, m in por_linha.items():
            if idx < len(linhas) and m.valor_final:
                rotulo = m.campo.rotulo
                linhas[idx] = f"{rotulo}: {m.valor_final}"
        resultado = "\n".join(linhas) + ("\n" if linhas else "")
        if not dry_run:
            saida.write_text(resultado, encoding="utf-8")
        for m in mapas:
            if m.precisa_revisao and not m.valor_final:
                m.status = StatusCampo.REVISAO
            elif m.valor_final:
                m.status = StatusCampo.PREENCHIDO if not dry_run else StatusCampo.OK
        return saida


if __name__ == "__main__":  # smoke-test do motor semântico
    logging.basicConfig(level=logging.INFO)
    sim = Similarizador()
    print("backend:", sim.backend)
    testes = [("Nome completo", "nome_completo"), ("Endereço", "address"),
              ("CPF", "Cadastro de Pessoas Físicas"), ("Telefone", "e-mail")]
    for a, b in testes:
        print(f"{a!r} x {b!r} = {sim.similaridade(a, b):.2f}")
