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

4. SIMULAÇÃO DE IA LOCAL (`extrair_com_ia`):
   Camada final que recebe o texto bruto e devolve um JSON com todos os pares
   encontrados. A implementação atual é determinística (baseada em sintaxe:
   duas-pontos, capitalização, terminação pontual), mas a assinatura foi
   desenhada para ser trocada por um LLM real (Ollama, llama.cpp, etc.) sem
   alterar os consumidores: basta substituir o corpo da função pela chamada
   HTTP/transformers e fazer `json.loads()` da resposta.

Saída unificada: lista de `DadoExtraido(chave_original, valor, fonte, confianca)`
ordenada por página/posição de leitura, pronta para o mapeamento semântico do
módulo `filler.py`.
"""

from __future__ import annotations

import json
import logging
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
    TXT_PONTO = "txt (rótulo: valor)"


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
    textuais; se nada produzir resultado suficiente, aciona `extrair_com_ia`.
    """

    def __init__(self, senha: Optional[str] = None) -> None:
        """
        Args:
            senha: Senha opcional para PDFs criptografados.
        """
        self.senha = senha

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

        # --- Estratégia 4: se as estruturas falharam, aciona a IA local ---
        if len(resultado.dados) < 3 and resultado.texto_bruto.strip():
            ia = extrair_com_ia(resultado.texto_bruto)
            if ia:
                resultado.avisos.append("Extração estruturada insuficiente; IA local acionada.")
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

    def __init__(self, encoding: str = "utf-8") -> None:
        """Args: encoding de leitura (padrão UTF-8, com fallback latin-1)."""
        self.encoding = encoding

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

        if len(resultado.dados) < 3 and texto.strip():
            ia = extrair_com_ia(texto)
            if ia:
                resultado.avisos.append("Extração textual insuficiente; IA local acionada.")
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
# 3) Camada de simulação de IA local
# ===========================================================================
def extrair_com_ia(texto_bruto: str) -> list[DadoExtraido]:
    """
    Chamada (simulada) a um LLM local para extração total.

    COMO FUNCIONA HOJE: implementa deterministicamente o comportamento esperado
    de um prompt de extração — percorre as linhas, reconhece pares por sinais
    sintáticos (dois-pontos, capitalização de rótulo, terminação) e devolve o
    MESMO formato de saída que um modelo real retornaria em JSON.

    COMO PLUGAR UM LLM REAL (Ollama/llama.cpp/HF transformers):
        payload = {"model": "llama3", "prompt": PROMPT_EXTRACAO + texto_bruto}
        resp = requests.post("http://localhost:11434/api/generate", json=payload)
        dados = json.loads(resp.json()["response"])  # lista de dicts
        return [DadoExtraido(**item) for item in dados]

    Args:
        texto_bruto: conteúdo textual integral do documento.

    Returns:
        Lista de `DadoExtraido` com fonte=IA_LOCAL e confiança moderada (0.65),
        pois saídas de modelo exigem revisão humana na GUI.
    """
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


PROMPT_EXTRACAO: str = (
    "Você é um motor de Document AI. Extraia TODOS os pares chave-valor deste "
    "documento, incluindo tabelas e relações espaciais. Responda APENAS com JSON: "
    '[{"chave_original": str, "valor": str, "fonte": "ia", "confianca": float}].\n\n---\n'
)


# ===========================================================================
# API de alto nível
# ===========================================================================
def extrair_documento(caminho: str | Path, senha: Optional[str] = None) -> ResultadoExtracao:
    """
    Ponto de entrada único: escolhe o extrator conforme a extensão do arquivo.

    Args:
        caminho: `.pdf` ou `.txt`.
        senha: senha opcional para PDFs criptografados.

    Returns:
        `ResultadoExtracao` consolidado.

    Raises:
        ValueError: extensão não suportada.
        FileNotFoundError/PermissionError/RuntimeError: ver extratores internos.
    """
    p = Path(caminho)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return ExtratorPDF(senha=senha).extrair(p)
    if ext == ".txt":
        return ExtratorTXT().extrair(p)
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
