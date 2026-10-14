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

## 🤖 IA Qwen — análise do documento e encaixe no formulário
O sistema usa o **Qwen** para ler TODO o conteúdo do PDF/TXT e decidir **onde
cada informação se encaixa** no formulário de destino, adaptando formatos
(data, CPF, endereço quebrado em rua/número/cidade) e devolvendo uma
**justificativa** por decisão.

Configuração (qualquer uma funciona; a GUI também tem o botão **🤖 IA Qwen**):
```bash
# 1) Ollama local (recomendado, offline)
ollama serve && ollama pull qwen2.5:7b-instruct
python gui.py                       # detecta http://localhost:11434 sozinho

# 2) API OpenAI-compatível (vLLM, LM Studio, Groq, OpenRouter...)
export QWEN_BASE_URL="https://seu-endpoint/v1"
export QWEN_API_KEY="sua-chave"
export QWEN_MODEL="qwen2.5-7b-instruct"
```
Como funciona o fluxo:
1. `extractor.extrair_documento(..., rotulos_destino=[...])` envia o texto do
   documento + a lista de campos do formulário ao Qwen (`consultar_qwen_encaixe`).
2. O modelo responde um JSON `[{"rotulo_destino", "chave_origem", "valor",
   "justificativa", "confianca"}]` → `ResultadoExtracao.encaixes_qwen`.
3. `filler.MapeadorSemanticos.mapear(campos, dados, encaixes_ia)` aplica em
   **duas camadas**: encaixes do Qwen primeiro (com confiança/justificativa da
   IA); os campos que a IA não cobriu caem na heurística fuzzy/neural.
4. Na GUI, cada linha do painel direito mostra a tag `[🤖 Qwen]`,
   `[🔎 heurística]` ou `[✋ manual]` + a justificativa; use **🧠 Reanalisar
   com IA** após carregar/editar, e o dropdown para corrigir qualquer erro.
Sem backend configurado, tudo continua funcionando via heurística determinística.

## Observações técnicas
* PDFs protegidos por senha levantam `PermissionError`; a GUI pede a senha.
* Sites fora do ar levantam `ConnectionError` tratado na GUI.
* Para PDFs de destino "visuais" (sem AcroForm), atribua `coordenadas=(x0, top, x1, bot)`
  aos `CampoDestino` (ex.: obtidas pela própria análise espacial do `extractor`).
* `sentence-transformers` é opcional: sem ele (ou offline), o sistema usa fuzzy matching.
