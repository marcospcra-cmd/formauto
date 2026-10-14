"""
extractor.py — Módulo de "Extração Total" de dados de documentos (PDF/TXT).
================================================================================

ARQUITETURA DA EXTRAÇÃO (por que este módulo NÃO usa listas fixas de regex):
----------------------------------------------------------------------------
Em vez de procurar apenas campos pré-definidos (CPF, Nome, etc.), o sistema
tenta capturar TODAS as informações do documento através de 4 estratégias
complementares, executadas em cascata:

1. TABELAS ESTRUTURAIS (pdfplumber.extract_tables):
   Linhas de tabela viram pares chave-valor automaticamente (coluna 0 = chave,
   demais colunas = valores). Confiança alta (0.95) pois a estrutura é explícita.

2. ANÁLISE ESPACIAL (pdfplumber.words + coordinates):
   Palavras são agrupadas em "linhas lógicas" por proximidade no eixo Y.
   Dentro de cada linha, detectamos dois padrões:
     a) Padrão "Rótulo: Valor"  -> separadores ":" ou "-" na mesma linha.
     b) Padrão "Colunas"         -> um bloco compacto à esquerda (CHAVE) e um
        bloco mais longo à direita (VALOR), MESMO SEM DOIS-PONTOS. A confiança
        é calculada dinamicamente pela coerência do layout (gap horizontal,
        alinhamento vertical, proporcionalidade dos blocos).

3. HEURÍSTICA DE LINHAS CONSECUTIVAS (fallback):
   Quando a análise espacial falha (ex.: PDF gerado sem coordenadas confiáveis),
   assumimos o padrão "linha N = chave / linha N+1 = valor" para linhas curtas
   seguidas de linhas um pouco maiores — comportamento comum em formulários
   impressos linearizados.

4. IA QWEN VIA NAVEGADOR (`extrair_com_ia` / `consultar_qwen_encaixe`):
   Camada final (e preferencial quando ativa). SEM NENHUMA CHAVE DE API: o
   módulo `qwen_navegador` abre o chat.qwen.ai no Chromium com um perfil
   persistente e usa a CONTA LOGADA do usuário para enviar o texto bruto do
   documento; o Qwen devolve um JSON com todos os pares encontrados e/ou já
   encaixados nos rótulos do formulário de destino. Se o navegador estiver
   sem sessão ou o site falhar, degradamos para uma heurística determinística
   local que devolve o MESMO formato de dados (extração continua offline).

Saída unificada: lista de `DadoExtraido(chave_original, valor, fonte, confianca)`
ordenada por página/posição de leitura, pronta para o mapeamento semântico do
módulo `filler.py`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("extractor")

# ---------------------------------------------------------------------------
# Constantes de calibração da análise espacial (ajustáveis por tipo de doc.)
# ---------------------------------------------------------------------------
TOL_Y: float = 3.0            # Tolerância (pt) para considerar palavras na mesma linha
MIN_KEY_LEN: int = 2          # Menor tamanho aceitável para uma "chave"
MAX_KEY_LEN: int = 60         # Maior tamanho aceitável para uma "chave"
MIN_VAL_LEN: int = 1          # Menor tamanho aceitável para um "valor"
KEY_MAX_WORDS: int = 5        # Chaves raramente têm mais que N palavras


class Fonte(str, Enum):
    """Origem estrutural de cada dado extraído (útil para depuração/UI)."""
    TABELA = "tabela"
    ESPACIAL_PONTO = "espacial (rótulo: valor)"
    ESPACIAL_COLUNAS = "espacial (colunas)"
    TEXTO_LINHAS = "heurística linhas consecutivas"
    IA_LOCAL = "simulação de IA local"
    IA_QWEN = "IA Qwen (encaixe no formulário)"
    TXT_PONTO = "txt (rótulo: valor)"


@dataclass
class EncaixeQwen:
    """
    Decisão de ENCAIXE produzida pelo Qwen: onde um dado do documento deve
    ser colocado em qual campo do formulário de destino.

    Attributes:
        rotulo_destino: Nome do campo do formulário (exato, como enviado).
        chave_origem: Rótulo/fragmento do PDF que contém o dado.
        valor: Conteúdo a gravar no campo (já adaptado ao formato do alvo).
        justificativa: Explicação curta da IA (aparece na GUI).
        confianca: 0..1 atribuído pelo modelo.
    """
    rotulo_destino: str
    chave_origem: str
    valor: str
    justificativa: str = ""
    confianca: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        """Serialização JSON-friendly."""
        return asdict(self)


@dataclass
class DadoExtraido:
    """
    Par chave-valor capturado de um documento de origem.

    Attributes:
        chave_original: Rótulo tal como aparece no documento (normalizado em
            caixa baixa, sem dois-pontos).
        valor: Conteúdo associado à chave.
        fonte: Estratégia que produziu o par (ver `Fonte`).
        confianca: Score 0..1 estimulado pela qualidade da evidência estrutural.
        pagina: Página de origem (None para TXT).
        posicao: Coordenadas aproximadas (x0, y0) para ordenação/revisão visual.
    """
    chave_original: str
    valor: str
    fonte: str
    confianca: float
    pagina: Optional[int] = None
    posicao: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializa o objeto para dicionário simples (JSON-friendly)."""
        d = asdict(self)
        return d


@dataclass
class ResultadoExtracao:
    """Contêiner da extração completa de um documento."""
    origem: str
    dados: list[DadoExtraido] = field(default_factory=list)
    texto_bruto: str = ""
    avisos: list[str] = field(default_factory=list)
    # Decisões de encaixe do Qwen (campo do formulário -> valor do PDF).
    # Preenchido apenas quando um backend de IA real está configurado e
    # `rotulos_destino` foi informado ao extrator.
    encaixes_qwen: list[EncaixeQwen] = field(default_factory=list)

    @property
    def como_dic(self) -> dict[str, str]:
        """Vista 'dicionário' dos dados (última ocorrência vence em duplicatas)."""
        out: dict[str, str] = {}
        for d in self.dados:
            out[d.chave_original] = d.valor
        return out


# ---------------------------------------------------------------------------
# Utilidades de normalização
# ---------------------------------------------------------------------------
def _normalizar_chave(texto: str) -> str:
    """Limpa um rótulo candidato: remove dois-pontos, espaços múltiplos, acentos de controle."""
    t = texto.strip().strip(":：-–—").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _eh_chave_plausivel(texto: str) -> bool:
    """
    Heurística GENÉRICA de plausibilidade de chave (não lista campos conhecidos!).
    Uma chave típica de formulário: é curta, tem poucas palavras, não termina
    com ponto final e não é uma frase verbal completa.
    """
    t = _normalizar_chave(texto)
    if not (MIN_KEY_LEN <= len(t) <= MAX_KEY_LEN):
        return False
    palavras = t.split()
    if len(palavras) > KEY_MAX_WORDS:
        return False
    if t.endswith("."):  # frases completas geralmente não são rótulos
        return False
    return True


def _alinhar_caixa(texto: str) -> str:
    """Preserva ALL CAPS como indício de rótulo, mas padroniza espaços."""
    return re.sub(r"\s+", " ", texto).strip()


# ===========================================================================
# 1) Extração de PDF
# ===========================================================================
class ExtratorPDF:
    """
    Extrator "total" para arquivos PDF usando pdfplumber.

    Combina `extract_tables()`, análise espacial de palavras e fallbacks
    textuais; se nada produzir resultado suficiente, aciona a IA (Qwen).

    IMPORTANTE — modo "IA sempre ativa": quando um backend de IA está
    configurado (`ConfigIAQwen.carregar()`), o Qwen passa a ser chamado em
    TODAS as extrações, não apenas como fallback. Ele lê o texto bruto do
    PDF E a lista de rótulos dos campos do formulário de destino, decidindo
    onde cada informação do documento deve ser encaixada/adaptada.
    """

    def __init__(self, senha: Optional[str] = None,
                 usar_ia: bool = True,
                 rotulos_destino: Optional[list[str]] = None) -> None:
        """
        Args:
            senha: Senha opcional para PDFs criptografados.
            usar_ia: Se True (padrão), consulta o LLM configurado (Qwen via
                Ollama / OpenAI-compatível / API Groq-OpenRouter). Sem
                configuração, cai na heurística determinística local.
            rotulos_destino: Rótulos dos campos do formulário de destino.
                Quando presentes, são enviados ao Qwen no prompt de
                ENCAIXE para que a IA já devolva os pares prontos para o
                formulário (com justificativa e confiança).
        """
        self.senha = senha
        self.usar_ia = usar_ia
        self.rotulos_destino = list(rotulos_destino or [])

    def extrair(self, caminho: str | Path) -> ResultadoExtracao:
        """
        Executa todas as estratégias sobre o PDF e consolida os resultados.

        Raises:
            FileNotFoundError: se o arquivo não existir.
            PermissionError: se o PDF estiver protegido e a senha faltar/incorreta.
            RuntimeError: para erros internos do pdfplumber.
        """
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"PDF não encontrado: {caminho}")

        resultado = ResultadoExtracao(origem=str(caminho))
        try:
            import pdfplumber  # import tardio: dependência pesada/opcional
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instale 'pdfplumber' para ler PDFs: pip install pdfplumber") from exc

        try:
            with pdfplumber.open(str(caminho), password=self.senha or "") as pdf:
                if getattr(pdf, "metadata", None) is None and pdf.pages == []:
                    raise PermissionError("PDF parece protegido por senha.")
                todas_palavras: list[dict[str, Any]] = []
                for i, pagina in enumerate(pdf.pages, start=1):
                    # --- Estratégia 1: tabelas estruturais ---
                    self._extrair_tabelas(pagina, i, resultado)
                    # --- Estratégia 2: análise espacial ---
                    palavras = pagina.extract_words(use_text_flow=False, keep_blank_chars=False)
                    for w in palavras:
                        w["_page"] = i
                    todas_palavras.extend(palavras)
                    espaciais = self._analise_espacial(palavras, i)
                    resultado.dados.extend(espaciais)
                resultado.texto_bruto = "\n".join(
                    (p or "") for p in (pg.extract_text() for pg in self._reabrir_textos(caminho))
                )
        except PermissionError as exc:
            raise PermissionError(
                "PDF protegido por senha. Forneça a senha no parâmetro 'senha'."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - fronteira de robustez
            msg = f"Falha ao ler PDF ({type(exc).__name__}): {exc}"
            logger.exception(msg)
            resultado.avisos.append(msg)

        # Deduplicação: mantém a versão de MAIOR confiança para cada par chave+valor
        resultado.dados = self._deduplicar(resultado.dados)

        # --- Estratégia 4: IA (Qwen) — SEMPRE ativa quando configurada ---
        # O modo PADRÃO é o navegador com a conta logada do Qwen (sem API).
        # O Qwen analisa o PDF em todas as extrações e decide onde cada
        # informação se encaixa no formulário. Se o navegador falhar (login
        # ausente, site fora do ar), degradamos para a heurística local.
        ia_sempre = self.usar_ia and bool(ConfigIAQwen.carregar())
        if self.usar_ia and resultado.texto_bruto.strip():
            # `encaixes` recebe as decisões do Qwen quando há formulário alvo;
            # None significa "backend indisponível/não configurado".
            encaixes: Optional[list[EncaixeQwen]] = None
            if ia_sempre and self.rotulos_destino:
                encaixes = consultar_qwen_encaixe(resultado.texto_bruto,
                                                  self.rotulos_destino)
                if encaixes is None:
                    resultado.avisos.append(
                        "Qwen via navegador indisponível (faça login no botão "
                        "'🤖 Qwen'); seguindo com extração estrutural.")
            estrutural_fraca = len(resultado.dados) < 3
            if encaixes is not None or estrutural_fraca:
                if encaixes is not None:
                    # Qwen respondeu: os pares já vêm com a chave nomeada igual ao
                    # rótulo do formulário => o mapeamento semântico converge ~100%.
                    resultado.encaixes_qwen = encaixes
                    ia = [DadoExtraido(e.rotulo_destino, e.valor, Fonte.IA_QWEN.value,
                                       max(0.5, min(0.98, e.confianca)))
                          for e in encaixes]
                    resultado.avisos.append(
                        f"Qwen analisou o PDF e propôs encaixe para {len(encaixes)} "
                        f"campo(s) do formulário.")
                else:
                    ia = extrair_com_ia(resultado.texto_bruto,
                                        rotulos_destino=self.rotulos_destino)
                    if ia:
                        resultado.avisos.append(
                            "IA (Qwen) consultada sobre todo o documento."
                            if ia_sempre else
                            "Extração estruturada insuficiente; IA local (simulada) acionada.")
                # A IA é a ÚLTIMA instância da cascata de confiança: ela só
                # substitui pares cuja evidência estrutural era mais fraca que
                # a sua (evita duplicatas e preserva o melhor candidato).
                por_chave = {d.chave_original.lower(): d for d in resultado.dados}
                for novo in ia:
                    atual = por_chave.get(novo.chave_original.lower())
                    if atual is None:
                        resultado.dados.append(novo)
                    elif novo.confianca > atual.confianca:
                        atual.chave_original, atual.valor = novo.chave_original, novo.valor
                        atual.fonte, atual.confianca = novo.fonte, novo.confianca
            elif self.usar_ia and not ia_sempre:
                resultado.avisos.append(
                    "IA desligada na configuração — usando apenas extração "
                    "estrutural. Ative o Qwen pelo navegador no botão '🤖 Qwen'.")

        resultado.dados = self._deduplicar(resultado.dados)

        resultado.dados.sort(key=lambda d: (d.pagina or 0,
                                            (d.posicao or (0.0, 0.0))[1],
                                            (d.posicao or (0.0, 0.0))[0]))
        return resultado

    # -- helpers internos ----------------------------------------------------
    def _reabrir_textos(self, caminho: Path):
        """Itera páginas apenas para extract_text (mantém código legível)."""
        import pdfplumber
        with pdfplumber.open(str(caminho), password=self.senha or "") as pdf:
            yield from pdf.pages

    @staticmethod
    def _extrair_tabelas(pagina: Any, n_pagina: int, resultado: ResultadoExtracao) -> None:
        """Estratégia 1: cada linha de tabela vira chave(coluna0)->valores(resto)."""
        try:
            tabelas = pagina.extract_tables()
        except Exception:  # noqa: BLE001 - nem todo PDF expõe linhas de tabela
            return
        for tabela in tabelas or []:
            for linha in tabela:
                celulas = [(_normalizar_chave(c) if c else "") for c in linha]
                celulas = [c for c in celulas if c]
                if len(celulas) >= 2 and _eh_chave_plausivel(celulas[0]):
                    resultado.dados.append(DadoExtraido(
                        chave_original=celulas[0],
                        valor=" | ".join(celulas[1:]),
                        fonte=Fonte.TABELA.value,
                        confianca=0.95,
                        pagina=n_pagina,
                    ))

    @staticmethod
    def _agrupar_linhas(palavras: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Agrupa palavras em linhas lógicas por proximidade no eixo Y (top)."""
        if not palavras:
            return []
        ordenadas = sorted(palavras, key=lambda w: (round(w["top"] / (TOL_Y + 1)), w["x0"]))
        linhas: list[list[dict[str, Any]]] = []
        atual: list[dict[str, Any]] = []
        ref_top: Optional[float] = None
        for w in ordenadas:
            if ref_top is None or abs(w["top"] - ref_top) <= TOL_Y:
                atual.append(w)
                ref_top = w["top"] if ref_top is None else (ref_top + w["top"]) / 2.0
            else:
                linhas.append(sorted(atual, key=lambda x: x["x0"]))
                atual = [w]
                ref_top = w["top"]
        if atual:
            linhas.append(sorted(atual, key=lambda x: x["x0"]))
        return linhas

    @staticmethod
    def _blocos_por_gap(linha: list[dict[str, Any]], gap_min: float = 18.0) -> list[list[dict[str, Any]]]:
        """Divide uma linha em blocos quando o espaço horizontal entre palavras excede `gap_min`."""
        blocos: list[list[dict[str, Any]]] = []
        atual: list[dict[str, Any]] = []
        for w in linha:
            if atual and (w["x0"] - atual[-1]["x1"]) > gap_min:
                blocos.append(atual)
                atual = [w]
            else:
                atual.append(w)
        if atual:
            blocos.append(atual)
        return blocos

    @classmethod
    def _analise_espacial(cls, palavras: list[dict[str, Any]], n_pagina: int) -> list[DadoExtraido]:
        """
        Estratégia 2: transforma geometria do layout em pares chave-valor.

        Regra central: se numa mesma linha lógica existe um bloco compacto à
        esquerda e outro bloco à direita, tratamos esquerda=CHAVE e direita=VALOR
        mesmo sem dois-pontos. A confiança cresce com:
          * existência de separador ":" no bloco esquerdo (+0.35);
          * bloco esquerdo plausível como rótulo (+0.15);
          * gap horizontal grande e limpo (+0.15);
          * valor visivelmente maior que a chave (+0.10).
        """
        saida: list[DadoExtraido] = []
        for linha in cls._agrupar_linhas(palavras):
            if not linha:
                continue
            texto_linha = " ".join(w["text"] for w in linha)
            top = linha[0]["top"]
            x0 = linha[0]["x0"]

            # 2a) Padrão "Rótulo: Valor" dentro da própria linha
            m = re.match(r"^\s*([^:]{2,60}?)\s*:\s*(.+)$", texto_linha)
            if m and _eh_chave_plausivel(m.group(1)):
                saida.append(DadoExtraido(
                    chave_original=_normalizar_chave(m.group(1)),
                    valor=m.group(2).strip(),
                    fonte=Fonte.ESPACIAL_PONTO.value,
                    confianca=0.90,
                    pagina=n_pagina,
                    posicao=(x0, top),
                ))
                continue

            # 2b) Padrão colunar sem dois-pontos
            blocos = cls._blocos_por_gap(linha)
            if len(blocos) >= 2:
                esq = " ".join(w["text"] for w in blocos[0])
                dir_ = " ".join(w["text"] for w in blocos[-1])
                gap = blocos[-1][0]["x0"] - blocos[0][-1]["x1"]
                if _eh_chave_plausivel(esq) and len(dir_.strip()) >= MIN_VAL_LEN and gap >= 12:
                    conf = 0.40
                    if gap >= 40:
                        conf += 0.15
                    if len(dir_) > len(esq):
                        conf += 0.10
                    if esq.isupper() or esq.endswith(("?", ")")) is False and len(esq.split()) <= 3:
                        conf += 0.10
                    # penaliza chaves que parecem frases verbais
                    if re.search(r"\b(de|da|do|em|para|com|que)\b", esq.lower()) and len(esq.split()) > 3:
                        conf -= 0.15
                    saida.append(DadoExtraido(
                        chave_original=_normalizar_chave(esq),
                        valor=dir_.strip(),
                        fonte=Fonte.ESPACIAL_COLUNAS.value,
                        confianca=max(0.30, min(conf, 0.85)),
                        pagina=n_pagina,
                        posicao=(x0, top),
                    ))
        return saida

    @staticmethod
    def _deduplicar(dados: list[DadoExtraido]) -> list[DadoExtraido]:
        """Mantém, para cada chave, o valor da estratégia mais confiável."""
        melhor: dict[str, DadoExtraido] = {}
        for d in dados:
            k = d.chave_original.lower()
            if k not in melhor or d.confianca > melhor[k].confianca:
                melhor[k] = d
        return list(melhor.values())


# ===========================================================================
# 2) Extração de TXT
# ===========================================================================
class ExtratorTXT:
    """Extrator para arquivos de texto puro: suporta 'chave: valor' e linhas alternadas."""

    def __init__(self, encoding: str = "utf-8", usar_ia: bool = True,
                 rotulos_destino: Optional[list[str]] = None) -> None:
        """
        Args:
            encoding: codificação de leitura (padrão UTF-8, fallback latin-1).
            usar_ia: consulta o Qwen quando configurado (mesma política do PDF).
            rotulos_destino: rótulos do formulário alvo para o prompt de encaixe.
        """
        self.encoding = encoding
        self.usar_ia = usar_ia
        self.rotulos_destino = list(rotulos_destino or [])

    def extrair(self, caminho: str | Path) -> ResultadoExtracao:
        """Lê o TXT e aplica as estratégias textuais + fallback de IA local."""
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo TXT não encontrado: {caminho}")
        try:
            texto = caminho.read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            texto = caminho.read_text(encoding="latin-1")

        resultado = ResultadoExtracao(origem=str(caminho), texto_bruto=texto)
        linhas = [l.rstrip() for l in texto.splitlines()]

        # Estratégia A: linhas "chave: valor" (ou "chave = valor")
        for idx, linha in enumerate(linhas):
            m = re.match(r"^\s*([^:=\n]{2,60}?)\s*[:=]\s*(.+?)\s*$", linha)
            if m and _eh_chave_plausivel(m.group(1)):
                resultado.dados.append(DadoExtraido(
                    chave_original=_normalizar_chave(m.group(1)),
                    valor=m.group(2).strip(),
                    fonte=Fonte.TXT_PONTO.value,
                    confianca=0.90,
                    posicao=(0.0, float(idx)),
                ))

        # Estratégia B: linhas consecutivas (curta=chave, próxima=valor)
        pares_b = self._heuristica_linhas_consecutivas(linhas)
        cobertas = {d.chave_original.lower() for d in resultado.dados}
        for par in pares_b:
            if par.chave_original.lower() not in cobertas:
                resultado.dados.append(par)

        # --- IA (Qwen): sempre que configurada (padrão = navegador logado) ---
        ia_sempre = self.usar_ia and bool(ConfigIAQwen.carregar())
        if self.usar_ia and texto.strip():
            encaixes: Optional[list[EncaixeQwen]] = None
            if ia_sempre and self.rotulos_destino:
                encaixes = consultar_qwen_encaixe(texto, self.rotulos_destino)
                if encaixes is None:
                    resultado.avisos.append(
                        "Qwen via navegador indisponível (faça login no botão "
                        "'🤖 Qwen'); seguindo com extração textual.")
            estrutural_fraca = len(resultado.dados) < 3
            if encaixes is not None or estrutural_fraca:
                if encaixes is not None:
                    resultado.encaixes_qwen = encaixes
                    ia = [DadoExtraido(e.rotulo_destino, e.valor, Fonte.IA_QWEN.value,
                                       max(0.5, min(0.98, e.confianca)))
                          for e in encaixes]
                    resultado.avisos.append(
                        f"Qwen analisou o TXT e propôs encaixe para {len(encaixes)} "
                        f"campo(s) do formulário.")
                else:
                    ia = extrair_com_ia(texto, rotulos_destino=self.rotulos_destino)
                    if ia:
                        resultado.avisos.append(
                            "IA (Qwen) consultada sobre todo o documento."
                            if ia_sempre else
                            "Extração textual insuficiente; IA local acionada.")
                # mesma política do PDF: a IA só substitui pares de confiança
                # inferior (é a última instância da cascata).
                por_chave = {d.chave_original.lower(): d for d in resultado.dados}
                for novo in ia:
                    atual = por_chave.get(novo.chave_original.lower())
                    if atual is None:
                        resultado.dados.append(novo)
                    elif novo.confianca > atual.confianca:
                        atual.chave_original, atual.valor = novo.chave_original, novo.valor
                        atual.fonte, atual.confianca = novo.fonte, novo.confianca

        return resultado

    @staticmethod
    def _heuristica_linhas_consecutivas(linhas: list[str]) -> list[DadoExtraido]:
        """
        Estratégia 3 (fallback): alterna linha-curta (rótulo) com linha-seguinte (valor).
        Funciona bem em formulários linearizados onde o valor fica abaixo do rótulo.
        """
        saida: list[DadoExtraido] = []
        i = 0
        while i < len(linhas) - 1:
            atual = _normalizar_chave(linhas[i])
            prox = linhas[i + 1].strip()
            if (atual and prox and _eh_chave_plausivel(atual)
                    and ":" not in atual and "=" not in atual
                    and len(atual) <= 40 and not _eh_chave_plausivel(prox) is False
                    and len(prox) >= len(atual) * 0.4):
                # evita engolir a próxima linha como chave dupla
                saida.append(DadoExtraido(
                    chave_original=atual,
                    valor=prox,
                    fonte=Fonte.TEXTO_LINHAS.value,
                    confianca=0.55,
                    posicao=(0.0, float(i)),
                ))
                i += 2
            else:
                i += 1
        return saida


# ===========================================================================
# 3) Camada de IA — Qwen via NAVEGADOR (conta logada, sem API) como padrão;
#    backends locais (Ollama / OpenAI-compatível) ficam opcionais.
# ===========================================================================
ARQ_CFG_IA: Path = Path.home() / ".autodoc_qwen.json"


@dataclass
class ConfigIAQwen:
    """
    Configuração do backend de IA (Qwen).

    MODOS SUPORTADOS (campo `estilo`):
      * "navegador"  -> MODO PADRÃO e RECOMENDADO. Nenhuma chave de API é
        usada: o módulo `qwen_navegador` abre o chat.qwen.ai no Chromium com
        um perfil persistente (`~/.autodoc_qwen_profile`) e conversa com o
        Qwen usando a CONTA LOGADA DO USUÁRIO no navegador. Login feito uma
        única vez reaproveita a sessão automaticamente.
      * "ollama"/"openai" -> modosopcionais para quem preferir rodar o modelo
        localmente (só ativados quando QWEN_BASE_URL/QWEN_MODEL apontam para
        um servidor local acessível; nunca exigem conta na nuvem).
      * None (carregar() devolve None) -> sem IA: cascata estrutural +
        heurística local continua funcionando 100% offline.

    Persistência: a escolha do usuário é gravada em `~/.autodoc_qwen.json`
    (via `salvar()`), com override por variáveis de ambiente.

    Attributes:
        estilo: "navegador" | "ollama" | "openai".
        base_url/api_key/modelo: usados apenas nos modos locais opcionais.
        timeout: segundos de espera máxima por análise.
    """
    estilo: str = "navegador"
    base_url: str = ""
    api_key: str = ""
    modelo: str = "Qwen (conta do navegador)"
    timeout: float = 240.0

    # -------------------------------------------------------- persistência
    @staticmethod
    def _ler_arquivo() -> dict[str, Any]:
        """Lê o JSON de configuração do usuário ({} se ausente/corrompido)."""
        try:
            dados = json.loads(ARQ_CFG_IA.read_text(encoding="utf-8"))
            return dados if isinstance(dados, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def salvar(self) -> None:
        """Grava esta configuração no arquivo de usuário (~/.autodoc_qwen.json)."""
        try:
            ARQ_CFG_IA.write_text(json.dumps(asdict(self), ensure_ascii=False,
                                             indent=2), encoding="utf-8")
        except OSError as exc:  # disco cheio/sem permissão: segue em memória
            logger.warning("Não foi possível salvar a config de IA: %s", exc)

    @classmethod
    def carregar(cls) -> Optional["ConfigIAQwen"]:
        """
        Resolve o backend ativo, nesta prioridade:

        1. AUTODOC_IA_OFF=1           -> None (IA desligada explicitamente).
        2. QWEN_BASE_URL no ambiente  -> modo local opcional (ollama/openai).
        3. Arquivo ~/.autodoc_qwen.json -> escolha salva pelo usuário na GUI
           ("navegador" => sempre ativo; "local" => sonda o Ollama).
        4. Sem nada configurado       -> MODO PADRÃO: navegador com a conta
           logada do Qwen (funciona mesmo sem arquivo de config).

        Returns:
            ConfigIAQwen ativa, ou None somente quando a IA foi desligada.
        """
        global _CFG_CACHE, _CFG_RESOLVIDO
        if _CFG_RESOLVIDO:
            return _CFG_CACHE

        cfg: Optional[ConfigIAQwen] = None
        if os.environ.get("AUTODOC_IA_OFF", "") == "1":
            cfg = None
        else:
            salvo = cls._ler_arquivo()
            base = os.environ.get("QWEN_BASE_URL", "").strip().rstrip("/")
            if base:  # override explícito por API local (modo opcional)
                estilo = "ollama" if "11434" in base else "openai"
                cfg = cls(estilo=estilo, base_url=base,
                          api_key=os.environ.get("QWEN_API_KEY", ""),
                          modelo=os.environ.get("QWEN_MODEL", "") or "qwen2.5:7b")
            elif salvo.get("estilo") == "local":
                cfg = cls._detectar_local(salvo)
            elif salvo.get("estilo") == "nenhum":
                cfg = None
            else:  # "navegador" ou primeira execução => padrão do projeto
                cfg = cls(estilo="navegador",
                          modelo=salvo.get("modelo") or "Qwen (conta do navegador)")
        _CFG_CACHE = cfg
        _CFG_RESOLVIDO = True
        return _CFG_CACHE

    @staticmethod
    def _detectar_local(salvo: dict[str, Any]) -> Optional["ConfigIAQwen"]:
        """Sonda um backend LOCAL (Ollama) — usado só se o usuário escolher."""
        url = (salvo.get("base_url")
               or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        modelo = salvo.get("modelo") or "qwen2.5:7b-instruct"
        try:
            import requests
            r = requests.get(f"{url}/api/tags", timeout=1.5)
            if r.ok:
                nomes = [m.get("name", "") for m in r.json().get("models", [])]
                escolhido = modelo if any(n.startswith(modelo) for n in nomes) else next(
                    (n for n in nomes if "qwen" in n.lower()), modelo)
                logger.info("Ollama local detectado em %s (modelo=%s)", url, escolhido)
                return ConfigIAQwen(estilo="ollama", base_url=url,
                                    api_key="", modelo=escolhido)
        except Exception as exc:  # noqa: BLE001 - rede é falível por natureza
            logger.debug("Ollama indisponível: %s", exc)
        return None


_CFG_CACHE: Optional[ConfigIAQwen] = None
_CFG_RESOLVIDO: bool = False


def reset_cache_ia() -> None:
    """Invalida o cache de configuração de IA (usado em testes/reconfiguração)."""
    global _CFG_RESOLVIDO
    _CFG_RESOLVIDO = False


def _chamar_qwen(prompt: str, cfg: ConfigIAQwen) -> str:
    """
    Envia o prompt ao Qwen e devolve o texto cru da resposta.

    Roteamento pelo modo ativo (`cfg.estilo`):
      * "navegador" (PADRÃO do projeto, SEM API): delega ao módulo
        `qwen_navegador`, que abre o chat.qwen.ai no Chromium com perfil
        persistente e usa a CONTA LOGADA do usuário para conversar com o
        modelo — nenhuma chave de API ou servidor local é necessário.
      * "ollama": POST {base}/api/generate {"model","prompt","stream":false}.
      * "openai": POST {base}/v1/chat/completions (vLLM, LM Studio...).
    Os modos locais são opcionais e usam `requests` quando disponível;
    senão, `urllib` (stdlib) como fallback.

    Raises:
        RuntimeError: navegador/serve indisponível ou resposta malformada —
            o chamador degrada graciosamente para a heurística local.
    """
    if cfg.estilo == "navegador":
        # Import tardio: mantém o extractor utilizável mesmo sem Playwright.
        import qwen_navegador as qn
        motor_cfg = qn.ConfigQwenNavegador.carregar()
        motor_cfg.timeout_resposta = cfg.timeout
        motor = qn.MotorQwenNavegador(motor_cfg)
        return motor.perguntar(prompt)

    payload: dict[str, Any]
    if cfg.estilo == "ollama":
        url = f"{cfg.base_url}/api/generate"
        payload = {"model": cfg.modelo, "prompt": prompt, "stream": False}
    else:
        url = f"{cfg.base_url}/v1/chat/completions"
        payload = {
            "model": cfg.modelo,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    try:
        import requests  # type: ignore
        resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout)
        resp.raise_for_status()
        data = resp.json()
    except ImportError:  # urllib puro, sem dependências externas
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=cfg.timeout) as fh:
            data = json.loads(fh.read().decode())
    except Exception as exc:  # noqa: BLE001 - fronteira de robustez de rede
        raise RuntimeError(
            f"Falha ao contatar o Qwen em {url} ({type(exc).__name__}: {exc}). "
            "Verifique se o Ollama está rodando (`ollama serve`) ou se "
            "QWEN_BASE_URL/QWEN_API_KEY estão corretos."
        ) from exc

    if cfg.estilo == "ollama":
        return str(data.get("response", ""))
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Resposta do Qwen em formato inesperado: {data}") from exc


def _extrair_json(resposta: str) -> list[Any]:
    """
    Extrai a primeira lista JSON válida da resposta do modelo.

    LLMs costumam envolver o JSON em cercas ```json ... ``` ou texto extra;
    esta função localiza o primeiro '[' balanceado e faz json.loads tolerante.
    No modo navegador isso é ainda mais importante, pois o chat frequentemente
    responde com prosa + bloco de código.
    """
    try:
        import qwen_navegador as qn
        return qn.extrair_json_da_resposta(resposta)
    except ImportError:  # pragma: no cover - módulo sempre acompanha o app
        m = re.search(r"\[.*\]", resposta, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Nenhum JSON encontrado na resposta: {resposta[:200]!r}")
        return json.loads(m.group(0))


PROMPT_ENCAIXE_TEMPLATE: str = """\
Você é o motor de IA do AutoDoc. Abaixo estão (1) o CONTEÚDO EXTRAÍDO DE UM \
DOCUMENTO (PDF/TXT) e (2) a LISTA DE CAMPOS de um formulário de destino.

Sua tarefa: analisar TODAS as informações do documento e decidir ONDE cada \
informação deve ser ENCAIXADA no formulário, adaptando o formato quando \
necessário (ex.: datas para o padrão do campo, CPF apenas dígitos, quebrar \
"Endereço completo" em rua/número/cidade quando existirem campos separados).

REGRAS:
- Use APENAS dados presentes no documento. NUNCA invente valores.
- Se nenhum dado do documento corresponde a um campo, omita esse campo.
- "confianca" vai de 0 a 1 (sua certeza sobre o encaixe).
- Responda SOMENTE com um JSON válido, sem comentários nem cercas de código:
[
  {{"rotulo_destino": "<nome EXATO do campo do formulário>",
    "chave_origem": "<rótulo/fragmento do documento que contém o dado>",
    "valor": "<valor final já adaptado>",
    "justificativa": "<por que este dado encaixa aqui>",
    "confianca": 0.9}}
]

--- CAMPOS DO FORMULÁRIO ---
{campos}

--- DOCUMENTO ---
{documento}
"""


def consultar_qwen_encaixe(texto_documento: str,
                           rotulos_destino: list[str]) -> Optional[list[EncaixeQwen]]:
    """
    Pede ao Qwen que analise o documento e proponha o ENCAIXE dos dados nos
    campos do formulário de destino.

    Returns:
        Lista de `EncaixeQwen` (pode ser vazia se o modelo não achar nada),
        ou None se não houver backend configurado / a chamada falhar — nesse
        caso o chamador mantém apenas a extração estrutural/heurística.

    Nota sobre o modo navegador: quando o Qwen é acessado pela conta logada
    no navegador, uma janela Chromium visível abre em cada análise (é assim
    que o usuário acompanha o login e evita bloqueios de automação).
    """
    cfg = ConfigIAQwen.carregar()
    if cfg is None:
        return None
    # limita o documento para caber na janela do modelo (heurística simples)
    doc = texto_documento[:12000]
    campos = "\n".join(f"- {r}" for r in rotulos_destino)
    prompt = PROMPT_ENCAIXE_TEMPLATE.format(campos=campos, documento=doc)
    try:
        resposta = _chamar_qwen(prompt, cfg)
        itens = _extrair_json(resposta)
    except Exception as exc:  # noqa: BLE001 - degradação graciosa é essencial
        logger.warning("Qwen indisponível/falhou: %s", exc)
        return None
    saida: list[EncaixeQwen] = []
    for it in itens:
        try:
            if not isinstance(it, dict):
                continue
            rot = str(it.get("rotulo_destino", "")).strip()
            val = str(it.get("valor", "")).strip()
            if not rot or not val:
                continue
            conf = float(it.get("confianca", 0.7))
            saida.append(EncaixeQwen(rotulo_destino=rot,
                                     chave_origem=str(it.get("chave_origem", rot)).strip(),
                                     valor=val,
                                     justificativa=str(it.get("justificativa", "")).strip(),
                                     confianca=max(0.0, min(1.0, conf))))
        except (TypeError, ValueError):
            continue  # item malformado: ignora individualmente, não aborta tudo
    logger.info("Qwen (%s/%s) retornou %d encaixe(s).",
                cfg.estilo, cfg.modelo, len(saida))
    return saida


def extrair_com_ia(texto_bruto: str,
                   rotulos_destino: Optional[list[str]] = None) -> list[DadoExtraido]:
    """
    Camada de IA com duas implementações possíveis:

    1) QWEN VIA NAVEGADOR (padrão do projeto, quando a IA está ativa):
       Sem chave de API: o módulo `qwen_navegador` usa a CONTA LOGADA do
       usuário no chat.qwen.ai (perfil persistente do Chromium) para enviar
       TODO o texto do documento; o modelo devolve um JSON com todos os pares
       chave-valor encontrados (extração total, sem listas fixas). Modos
       locais opcionais (Ollama/OpenAI-compatível) permanecem suportados para
       quem preferir rodar o modelo na própria máquina.

    2) HEURÍSTICA DETERMINÍSTICA (fallback offline / IA desligada):
       Percorre as linhas reconhecendo pares por sinais sintáticos
       (dois-pontos, capitalização de rótulo, terminação) e devolve o MESMO
       formato que o modelo real retornaria — assim os consumidores
       (filler/GUI) funcionam igual sem GPU nem internet.

    Args:
        texto_bruto: conteúdo textual integral do documento.
        rotulos_destino: rótulos do formulário alvo (contexto extra p/ o Qwen).

    Returns:
        Lista de `DadoExtraido`; fonte=IA_QWEN quando o Qwen respondeu (pelo
        navegador ou backend local), fonte=IA_LOCAL na heurística.
    """
    cfg = ConfigIAQwen.carregar()
    if cfg is not None:
        prompt = PROMPT_EXTRACAO.format(
            campos=("\n".join(f"- {r}" for r in (rotulos_destino or [])) or "(nenhum)"),
            documento=texto_bruto[:12000])
        try:
            itens = _extrair_json(_chamar_qwen(prompt, cfg))
            saida_q: list[DadoExtraido] = []
            for it in itens:
                if not isinstance(it, dict):
                    continue
                chave = str(it.get("chave_original", "")).strip()
                valor = str(it.get("valor", "")).strip()
                if chave and valor:
                    saida_q.append(DadoExtraido(
                        chave, valor, Fonte.IA_QWEN.value,
                        max(0.0, min(1.0, float(it.get("confianca", 0.8)))),
                        posicao=(0.0, 0.0)))
            if saida_q:
                return saida_q
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chamada ao Qwen falhou (%s); usando heurística local.", exc)

    dados: list[DadoExtraido] = []
    linhas = [l.strip() for l in texto_bruto.splitlines() if l.strip()]
    for idx, linha in enumerate(linhas):
        # Sinal 1: separador explícito
        m = re.match(r"^([A-ZÀ-Ý][^:\n]{1,60}?)\s*:\s*(.+)$", linha)
        if m:
            chave, valor = _normalizar_chave(m.group(1)), m.group(2).strip()
            if chave and valor:
                dados.append(DadoExtraido(chave, valor, Fonte.IA_LOCAL.value, 0.70,
                                         posicao=(0.0, float(idx))))
            continue
        # Sinal 2: rótulo isolado seguido de conteúdo (duas linhas)
        if (idx + 1 < len(linhas) and _eh_chave_plausivel(linha)
                and linha[:1].isupper() and len(linha) <= 40 and ":" not in linha):
            valor = linhas[idx + 1]
            if valor and valor != linha:
                dados.append(DadoExtraido(_normalizar_chave(linha), valor,
                                         Fonte.IA_LOCAL.value, 0.60, posicao=(0.0, float(idx))))
    return dados


PROMPT_EXTRACAO: str = """\
Você é um motor de Document AI. Extraia TODOS os pares chave-valor deste \
documento, incluindo tabelas e relações espaciais. Se houver uma lista de \
campos de formulário abaixo, priorize chaves com esses mesmos nomes.

Campos do formulário (opcional):
{campos}

Responda APENAS com JSON válido, sem cercas de código:
[{{"chave_original": str, "valor": str, "fonte": "ia", "confianca": float}}]

--- DOCUMENTO ---
{documento}
"""


# ===========================================================================
# API de alto nível
# ===========================================================================
def extrair_documento(caminho: str | Path, senha: Optional[str] = None,
                      rotulos_destino: Optional[list[str]] = None,
                      usar_ia: bool = True) -> ResultadoExtracao:
    """
    Ponto de entrada único: escolhe o extrator conforme a extensão do arquivo.

    Args:
        caminho: `.pdf` ou `.txt`.
        senha: senha opcional para PDFs criptografados.
        rotulos_destino: rótulos dos campos do formulário alvo; quando há um
            backend Qwen configurado, são enviados à IA para que ela decida
            ONDE encaixar cada informação do documento.
        usar_ia: desabilite (False) para força apenas extração estrutural.

    Returns:
        `ResultadoExtracao` consolidado (inclui `encaixes_qwen` quando a IA
        real respondeu).

    Raises:
        ValueError: extensão não suportada.
        FileNotFoundError/PermissionError/RuntimeError: ver extratores internos.
    """
    p = Path(caminho)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return ExtratorPDF(senha=senha, usar_ia=usar_ia,
                           rotulos_destino=rotulos_destino).extrair(p)
    if ext == ".txt":
        return ExtratorTXT(usar_ia=usar_ia,
                           rotulos_destino=rotulos_destino).extrair(p)
    raise ValueError(f"Formato não suportado: '{ext}'. Use .pdf ou .txt.")


if __name__ == "__main__":  # smoke-test rápido pela CLI
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Uso: python extractor.py <arquivo.pdf|arquivo.txt> [senha]")
        sys.exit(1)
    res = extrair_documento(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps([d.to_dict() for d in res.dados], ensure_ascii=False, indent=2))
    for av in res.avisos:
        print("AVISO:", av)
