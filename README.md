# AutoDoc AI — Extração Inteligente e Preenchimento Automático de Formulários

Sistema modular em Python que lê documentos de origem (**PDF/TXT**), extrai
**todas** as informações (pares chave-valor, tabelas, relações espaciais) e
preenche destinos (**Formulário Web via Playwright, PDF via pypdf/reportlab, TXT**).

## Arquitetura (3 módulos)
| Arquivo | Responsabilidade |
|---|---|
| `extractor.py` | Leitura/estruturação bruta: tabelas (`pdfplumber.extract_tables`), análise espacial (chave à esquerda / valor à direita **sem dois-pontos**), heurística de linhas consecutivas e fallback `extrair_com_ia()` (simulação plugável de LLM local). Saída: `DadoExtraido(chave_original, valor, fonte, confianca)`. |
| `filler.py` | Mapeamento **semântico** (sentence-transformers `all-MiniLM-L6-v2`, com fallback automático para `thefuzz`) entre rótulos do destino e chaves da origem; limiar 0.75 → auto-mapeamento, abaixo disso → "revisão necessária". Executores: Web (coleta name/id/placeholder/aria-label/label; destaque verde/laranja), PDF (AcroForm + overlay visual por coordenadas com reportlab) e TXT. |
| `gui.py` | Tkinter/ttk moderno: painel esquerdo com tabela editável de todos os dados extraídos (duplo-clique corrige valores); painel direito com valor mapeado, barra de confiança e Combobox para reatribuição manual. Botões: Carregar Origem, Carregar Destino, Simular (Dry-run), Executar. |

## Instalação e uso
```bash
pip install -r requirements.txt
playwright install chromium        # apenas para destino Web
python gui.py                      # interface gráfica
# CLI de teste do extrator:
python extractor.py documento.pdf [senha]
```

## Observações técnicas
* PDFs protegidos por senha levantam `PermissionError`; a GUI pede a senha.
* Sites fora do ar levantam `ConnectionError` tratado na GUI.
* Para PDFs de destino "visuais" (sem AcroForm), atribua `coordenadas=(x0, top, x1, bot)`
  aos `CampoDestino` (ex.: obtidas pela própria análise espacial do `extractor`).
* `sentence-transformers` é opcional: sem ele (ou offline), o sistema usa fuzzy matching.
