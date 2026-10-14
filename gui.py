"""
gui.py — Interface gráfica de controle, revisão e mapeamento (Tkinter/ttk).
================================================================================

Layout em duas colunas sobre Notebook:
  * ABA "Revisão & Mapeamento":
      - PAINEL ESQUERDO (Origem): Treeview editável com TODOS os dados
        extraídos (chave, valor editável, fonte, confiança, página).
        Duplo-clique na coluna "Valor" abre edição inline.
      - PAINEL DIREITO (Destino): para cada campo do formulário alvo mostra
        o valor mapeado, uma barra de confiança (ttk.Progressbar) e um
        Combobox para reatribuir manualmente qualquer dado da origem.
  * BARRA SUPERIOR: Carregar Origem / Carregar Destino / Simular (Dry-run) /
    Executar Preenchimento + status geral.

Executores longos (Playwright/IA) rodam em threads separadas para nunca
congelar a UI; resultados voltam via `root.after` (thread-safe).
Execute:  python gui.py
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

from extractor import (
    ConfigIAQwen,
    DadoExtraido,
    EncaixeQwen,
    ResultadoExtracao,
    consultar_qwen_encaixe,
    extrair_documento,
    reset_cache_ia,
)
from filler import (
    CampoDestino,
    MapeadorSemanticos,
    MapeamentoCampo,
    PreenchedorPDF,
    PreenchedorTXT,
    PreenchedorWeb,
    Similarizador,
    StatusCampo,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("gui")

# Paleta "moderna" sóbria
COR_FUNDO = "#f4f5f7"
COR_PAINEL = "#ffffff"
COR_ACENTO = "#2f6fed"
COR_OK = "#1a7f37"
COR_WARN = "#b45309"
COR_ERRO = "#b91c1c"


class AplicacaoPreenchimento:
    """Classe-mestre que une extractor + filler + widgets Tk."""

    def __init__(self, root: tk.Tk) -> None:
        """Constrói a janela e inicializa o estado de negócio."""
        self.root = root
        self.resultado: Optional[ResultadoExtracao] = None
        self.campos_destino: list[CampoDestino] = []
        self.mapas: list[MapeamentoCampo] = []
        self.destino_url: Optional[str] = None
        self.destino_arquivo: Optional[Path] = None
        self.sim = Similarizador()
        self.mapeador = MapeadorSemanticos(self.sim)
        # Estado da IA Qwen: configuração detectada + último resultado bruto
        self.cfg_ia: Optional[ConfigIAQwen] = ConfigIAQwen.carregar()
        self._encaixes_brutos: list[EncaixeQwen] = []

        self._estilo()
        self._topo()
        self._corpo()
        self._set_status("Pronto. Carregue um documento de origem (.pdf/.txt).")

    # ------------------------------------------------------------------ estilo
    def _estilo(self) -> None:
        """Configura ttk 'clam' com paleta limpa e fontes legíveis."""
        self.root.title("AutoDoc AI — Extração Inteligente e Preenchimento Automático")
        self.root.geometry("1280x760")
        self.root.minsize(1000, 620)
        self.root.configure(bg=COR_FUNDO)
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure(".", background=COR_FUNDO, font=("Segoe UI", 10))
        st.configure("TNotebook", background=COR_FUNDO, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))
        st.configure("Card.TFrame", background=COR_PAINEL, relief="flat")
        st.configure("TLabel", background=COR_PAINEL)
        st.configure("TLabelframe", background=COR_PAINEL)
        st.configure("TLabelframe.Label", background=COR_PAINEL)
        st.configure("TCheckbutton", background=COR_FUNDO)
        st.configure("Titulo.TLabel", background=COR_PAINEL,
                     font=("Segoe UI", 11, "bold"), foreground="#111827")
        st.configure("TButton", padding=(10, 6), font=("Segoe UI", 10, "bold"))
        st.configure("Primario.TButton", foreground="white", background=COR_ACENTO)
        st.map("Primario.TButton", background=[("active", "#1e56d1")])
        st.configure("Treeview", rowheight=26, fieldbackground=COR_PAINEL,
                     background=COR_PAINEL)
        st.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        st.configure("Verde.Horizontal.TProgressbar", troughcolor="#e5e7eb",
                     background=COR_OK)
        st.configure("Laranja.Horizontal.TProgressbar", troughcolor="#e5e7eb",
                     background="#d97706")
        st.configure("Cinza.Horizontal.TProgressbar", troughcolor="#e5e7eb",
                     background="#9ca3af")

    # -------------------------------------------------------------------- topo
    def _topo(self) -> None:
        """Barra superior com os 4 botões de ação exigidos + status."""
        barra = ttk.Frame(self.root, style=None)
        barra.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Button(barra, text="📄 Carregar Origem",
                   command=self.carregar_origem).pack(side="left", padx=(0, 6))
        ttk.Button(barra, text="🎯 Carregar Destino",
                   command=self.carregar_destino).pack(side="left", padx=6)
        ttk.Button(barra, text="🧪 Simular Preenchimento (Dry-run)",
                   command=lambda: self._executar_preenchimento(dry_run=True)
                   ).pack(side="left", padx=6)
        ttk.Button(barra, text="🚀 Executar Preenchimento",
                   style="Primario.TButton",
                   command=lambda: self._executar_preenchimento(dry_run=False)
                   ).pack(side="left", padx=6)
        ttk.Separator(barra, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(barra, text="🤖 IA Qwen",
                   command=self._dialogo_ia).pack(side="left", padx=6)
        self.lbl_backend = ttk.Label(barra, text=f"Motor semântico: {self.sim.backend}",
                                     background=COR_FUNDO, foreground="#6b7280")
        self.lbl_ia = ttk.Label(barra, text="", background=COR_FUNDO,
                                foreground="#6b7280")
        self.lbl_ia.pack(side="right")
        self._atualizar_label_ia()
        self.lbl_backend.pack(side="right")

        self.status_var = tk.StringVar(value="")
        self.lbl_status = ttk.Label(self.root, textvariable=self.status_var,
                                    background=COR_FUNDO, anchor="w",
                                    font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(fill="x", padx=14, pady=(0, 4))

    # ------------------------------------------------------------------- corpo
    def _corpo(self) -> None:
        """Notebook com a aba principal de dois painéis."""
        self.note = ttk.Notebook(self.root)
        self.note.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        aba = ttk.Frame(self.note)
        self.note.add(aba, text="  Revisão & Mapeamento  ")

        paned = ttk.Panedwindow(aba, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)

        esq = ttk.Frame(paned, style="Card.TFrame")
        dir_ = ttk.Frame(paned, style="Card.TFrame")
        paned.add(esq, weight=1)
        paned.add(dir_, weight=1)

        # ---- Painel esquerdo: tabela editável da ORIGEM ----
        ttk.Label(esq, text="ORIGEM — dados extraídos (duplo-clique no valor p/ editar)",
                  style="Titulo.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
        cols = ("chave", "valor", "fonte", "conf", "pag")
        self.tv_origem = ttk.Treeview(esq, columns=cols, show="headings", height=18)
        larguras = {"chave": 170, "valor": 260, "fonte": 150, "conf": 60, "pag": 45}
        titulos = {"chave": "Chave", "valor": "Valor", "fonte": "Fonte",
                   "conf": "Conf.", "pag": "Pág"}
        for c in cols:
            self.tv_origem.heading(c, text=titulos[c])
            self.tv_origem.column(c, width=larguras[c], anchor="w")
        scroll_o = ttk.Scrollbar(esq, orient="vertical", command=self.tv_origem.yview)
        self.tv_origem.configure(yscrollcommand=scroll_o.set)
        self.tv_origem.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scroll_o.pack(side="right", fill="y", pady=10)
        self.tv_origem.bind("<Double-1>", self._editar_valor_origem)

        # ---- Painel direito: campos do DESTINO ----
        dir_wrap = ttk.Frame(dir_)
        dir_wrap.pack(fill="x", padx=10, pady=(0, 2))
        ttk.Label(dir_, text="DESTINO — campos do formulário e mapeamento",
                  style="Titulo.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
        self.chk_ia = tk.BooleanVar(value=True)
        ttk.Checkbutton(dir_wrap, text="🤖 Mostrar justificativas do Qwen",
                        variable=self.chk_ia, command=self._reatualizar_mapeamentos
                        ).pack(side="left")
        ttk.Button(dir_wrap, text="🧠 Reanalisar com IA",
                   command=self.reanalisar_com_ia).pack(side="right")
        self.frame_campos = ttk.Frame(dir_)
        canvas = tk.Canvas(dir_, highlightthickness=0, bg=COR_PAINEL)
        scroll_d = ttk.Scrollbar(dir_, orient="vertical", command=canvas.yview)
        self.frame_campos.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame_campos, anchor="nw")
        canvas.configure(yscrollcommand=scroll_d.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scroll_d.pack(side="right", fill="y", pady=10)
        # Largura dos comboboxes acompanha o painel
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(0, width=e.width - 20))

    # --------------------------------------------------------- ações de carga
    def carregar_origem(self) -> None:
        """Diálogo .pdf/.txt -> extrator completo -> preenche a tabela esquerda."""
        caminho = filedialog.askopenfilename(
            title="Documento de origem",
            filetypes=[("Documentos", "*.pdf *.txt"), ("PDF", "*.pdf"), ("TXT", "*.txt")])
        if not caminho:
            return
        senha: Optional[str] = None
        try:
            # Os rótulos do destino (se já carregado) vão para a IA decidir o
            # encaixe de cada informação do documento diretamente no formulário.
            rotulos = [c.rotulo for c in self.campos_destino] or None
            res = extrair_documento(caminho, senha=senha, rotulos_destino=rotulos)
        except PermissionError:
            senha = simpledialog_get_password(self.root, "PDF protegido",
                                              "Digite a senha do PDF:")
            if not senha:
                messagebox.showerror("Erro", "PDF exige senha.")
                return
            try:
                res = extrair_documento(caminho, senha=senha, rotulos_destino=rotulos)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Erro", f"Falha ao ler PDF: {exc}")
                return
        except Exception as exc:  # FileNotFoundError, ValueError...
            messagebox.showerror("Erro", str(exc))
            return

        self.resultado = res
        self._encaixes_brutos = list(res.encaixes_qwen)
        self._renderir_origem(res)
        aviso = ("  ⚠ " + "; ".join(res.avisos)) if res.avisos else ""
        self._set_status(f"{len(res.dados)} pares extraídos de {Path(caminho).name}.{aviso}")
        self._reatualizar_mapeamentos()

    def carregar_destino(self) -> None:
        """Escolhe Web (URL) ou arquivo (PDF/TXT template) como destino."""
        escolha = messagebox.askyesno(
            "Tipo de destino",
            "Sim = Formulário Web (URL)\nNão = Arquivo PDF/TXT (template)")
        if escolha is None:
            return
        if escolha:
            url = simpledialog_askstring(self.root, "URL do formulário",
                                         "https://exemplo.com/formulario")
            if not url:
                return
            self.destino_url, self.destino_arquivo = url, None
            self._set_status(f"Inspecionando {url} ...")
            threading.Thread(target=self._carregar_web_bg, args=(url,),
                             daemon=True).start()
        else:
            caminho = filedialog.askopenfilename(
                title="Template de destino",
                filetypes=[("Templates", "*.pdf *.txt"), ("PDF", "*.pdf"), ("TXT", "*.txt")])
            if not caminho:
                return
            self.destino_arquivo, self.destino_url = Path(caminho), None
            try:
                if caminho.lower().endswith(".pdf"):
                    self.campos_destino = PreenchedorPDF(caminho).coletar_campos()
                else:
                    self.campos_destino = PreenchedorTXT(caminho).coletar_campos()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Erro", f"Falha ao inspecionar destino: {exc}")
                return
            if not self.campos_destino:
                messagebox.showwarning(
                    "Destino visual",
                    "O PDF não tem campos AcroForm.\nPara preenchimento visual, "
                    "edite os valores na ORIGEM e use 'Executar'; coordenadas "
                    "podem ser adicionadas programaticamente via CampoDestino.")
            self._reatualizar_mapeamentos()
            self._set_status(f"{len(self.campos_destino)} campos de destino carregados.")

    def _carregar_web_bg(self, url: str) -> None:
        """Thread: coleta campos web sem travar a UI."""
        try:
            campos = PreenchedorWeb(headless=True).coletar_campos(url)
            self.root.after(0, self._on_web_campos, campos, url, None)
        except Exception as exc:  # ConnectionError, RuntimeError...
            self.root.after(0, self._on_web_campos, [], url, exc)

    def _on_web_campos(self, campos: list[CampoDestino], url: str,
                       erro: Optional[Exception]) -> None:
        """Callback main-thread para resultado da inspeção web."""
        if erro:
            messagebox.showerror("Destino Web", f"Não foi possível acessar:\n{erro}")
            self._set_status("Falha ao carregar destino web.")
            return
        self.campos_destino = campos
        self._reatualizar_mapeamentos()
        self._set_status(f"{len(campos)} campos web detectados em {url}.")

    # ------------------------------------------------------------ renderização
    def _renderir_origem(self, res: ResultadoExtracao) -> None:
        """Preenche a Treeview da origem com todos os DadoExtraido."""
        self.tv_origem.delete(*self.tv_origem.get_children())
        for d in res.dados:
            self.tv_origem.insert(
                "", "end", iid=d.chave_original,
                values=(d.chave_original, d.valor, d.fonte,
                        f"{d.confianca:.0%}", d.pagina or "-"))

    def _sincronizar_valores_ui(self) -> None:
        """Copia edições feitas na tabela para os objetos DadoExtraido."""
        if not self.resultado:
            return
        por_chave = {d.chave_original: d for d in self.resultado.dados}
        for iid in self.tv_origem.get_children():
            vals = self.tv_origem.item(iid, "values")
            chave, novo_valor = vals[0], vals[1]
            if chave in por_chave and por_chave[chave].valor != novo_valor:
                por_chave[chave].valor = novo_valor
                por_chave[chave].confianca = min(1.0, por_chave[chave].confianca)

    def _reatualizar_mapeamentos(self) -> None:
        """Roda o mapeador (Qwen em 1º lugar, heurística como fallback) e
        redesenha o painel direito."""
        self._sincronizar_valores_ui()
        for w in self.frame_campos.winfo_children():
            w.destroy()
        self.mapas = []
        if not self.campos_destino:
            ttk.Label(self.frame_campos,
                      text="Nenhum destino carregado. Use 'Carregar Destino'.",
                      foreground="#6b7280").pack(padx=12, pady=20)
            return
        dados = self.resultado.dados if self.resultado else []
        # Camada 1 = encaixes propostos pelo Qwen (se houver); camada 2 =
        # similaridade heurística apenas para os campos que a IA não cobriu.
        self.mapas = self.mapeador.mapear(self.campos_destino, dados,
                                          encaixes_ia=self._encaixes_brutos)
        for m in self.mapas:
            self._linha_campo(m, dados)

    def _linha_campo(self, m: MapeamentoCampo, dados: list[DadoExtraido]) -> None:
        """Uma linha do painel direito: rótulo, valor, barra de confiança, dropdown."""
        frame = ttk.Frame(self.frame_campos)
        frame.pack(fill="x", padx=10, pady=6)

        cor = COR_OK if m.status in (StatusCampo.OK, StatusCampo.PREENCHIDO) else (
            COR_WARN if m.status in (StatusCampo.REVISAO, StatusCampo.FALHA) else "#6b7280")
        tag = " ✔" if m.status in (StatusCampo.OK, StatusCampo.PREENCHIDO) else (
            " ⚠ revisão" if m.status == StatusCampo.REVISAO else " ✖ sem dado")
        origem = {"qwen": "🤖 Qwen", "manual": "✋ manual"}.get(m.origem_encaixe, "🔎 heurística")
        cab = ttk.Frame(frame)
        cab.grid(row=0, column=0, sticky="w")
        ttk.Label(cab, text=m.campo.rotulo + tag, foreground=cor,
                  font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(cab, text=f"  [{origem}]", foreground="#6b7280",
                  font=("Segoe UI", 8, "italic")).pack(side="left")
        ttk.Label(frame, text=f"confiança: {m.score:.0%}", foreground=cor
                  ).grid(row=0, column=2, sticky="e", padx=(8, 0))

        var = tk.StringVar(value=m.valor_final)
        ent = ttk.Entry(frame, textvariable=var, width=42)
        ent.grid(row=1, column=0, columnspan=2, sticky="we", pady=(2, 2))
        ent.bind("<FocusOut>", lambda e, mm=m, vv=var: self._ao_editar_valor(mm, vv.get()))

        pb = ttk.Progressbar(frame, length=120, mode="determinate",
                             value=int(m.score * 100))
        estilo_pb = ("Verde" if m.status in (StatusCampo.OK, StatusCampo.PREENCHIDO) else
                     "Laranja" if m.status in (StatusCampo.REVISAO, StatusCampo.FALHA) else "Cinza")
        pb.configure(style=f"{estilo_pb}.Horizontal.TProgressbar")
        pb.grid(row=2, column=0, sticky="w", pady=(0, 4))

        chaves = [d.chave_original for d in dados]
        cbv = tk.StringVar(value=(m.dado.chave_original if m.dado else ""))
        combo = ttk.Combobox(frame, values=["— nenhum —"] + chaves,
                             textvariable=cbv, state="readonly", width=28)
        combo.grid(row=2, column=1, columnspan=2, sticky="e", pady=(0, 4))
        combo.bind("<<ComboboxSelected>>",
                   lambda e, mm=m, cc=cbv, vv=var: self._remapear_manual(mm, cc.get(), vv))
        frame.columnconfigure(0, weight=1)

        # Justificativa do encaixe (Qwen ou heurística) — linha auxiliar cinza
        if m.justificativa and self.chk_ia.get():
            prefixo = "🤖 Qwen: " if m.origem_encaixe == "qwen" else "ℹ "
            ttk.Label(frame, text=prefixo + m.justificativa, foreground="#6b7280",
                      font=("Segoe UI", 8, "italic"), wraplength=520, justify="left"
                      ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 2))

    # ------------------------------------------------------------- interações
    def _editar_valor_origem(self, event: Any) -> None:
        """Edição inline da coluna Valor na tabela de origem (duplo clique)."""
        tv = self.tv_origem
        regiao = tv.identify_region(event.x, event.y)
        if regiao != "cell":
            return
        col_id = tv.identify_column(event.x)
        if col_id != "#2":  # apenas a coluna Valor é editável
            return
        iid = tv.identify_row(event.y)
        if not iid:
            return
        x, y, w, h = tv.bbox(iid, col_id)
        entry = ttk.Entry(tv)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, tv.set(iid, "valor"))
        entry.select_range(0, tk.END)
        entry.focus_set()

        def confirmar(_ev: Any = None) -> None:
            tv.set(iid, "valor", entry.get())
            entry.destroy()
            self._reatualizar_mapeamentos()

        entry.bind("<Return>", confirmar)
        entry.bind("<FocusOut>", confirmar)
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _ao_editar_valor(self, m: MapeamentoCampo, novo: str) -> None:
        """Usuário corrigiu manualmente o valor previsto de um campo destino."""
        if novo != m.valor_final:
            m.valor_final = novo
            m.origem_encaixe = "manual"
            if m.status in (StatusCampo.REVISAO, StatusCampo.SEM_DADO) and novo:
                m.status = StatusCampo.OK
                m.score = max(m.score, 0.80)
            m.justificativa = f"valor corrigido manualmente para '{novo[:60]}'"
            self._reatualizar_mapeamentos()

    # ----------------------------------------------------- IA Qwen: análise/reconfig
    def reanalisar_com_ia(self) -> None:
        """
        Reexecuta o ENCAIXE do Qwen sob demanda (botão '🧠 Reanalisar com IA').

        Útil quando o destino foi carregado DEPOIS da origem, ou quando o
        usuário corrigiu valores na tabela e quer nova opinião da IA. Roda em
        thread para não congelar a UI; ao voltar, injeta os encaixes como
        camada preferencial do mapeamento.
        """
        if not self.resultado or not self.campos_destino:
            messagebox.showwarning(
                "Reanálise", "Carregue a ORIGEM e o DESTINO antes de usar a IA.")
            return
        if not ConfigIAQwen.carregar():
            # IA desligada na configuração -> oferece religar pelo navegador.
            self._dialogo_ia()
            if not ConfigIAQwen.carregar():
                messagebox.showinfo(
                    "IA indisponível",
                    "A IA Qwen está desligada na configuração.\n"
                    "O sistema seguirá apenas com o mapeamento heurístico.")
                return
        texto = self.resultado.texto_bruto
        rotulos = [c.rotulo for c in self.campos_destino]
        self._set_status("🤖 Qwen (conta do navegador) analisando o documento e "
                         "propondo encaixes — aguarde a janela do chat ...")
        threading.Thread(target=self._reanalise_bg, args=(texto, rotulos),
                         daemon=True).start()

    def _reanalise_bg(self, texto: str, rotulos: list[str]) -> None:
        """Thread: consulta o Qwen sem bloquear a interface."""
        try:
            encaixes = consultar_qwen_encaixe(texto, rotulos)
            self.root.after(0, self._on_reanalise, encaixes, None)
        except Exception as exc:  # noqa: BLE001 - fronteira de robustez
            self.root.after(0, self._on_reanalise, None, exc)

    def _on_reanalise(self, encaixes: Optional[list[EncaixeQwen]],
                      erro: Optional[Exception]) -> None:
        """Callback main-thread: aplica os encaixes propostos pelo Qwen."""
        if erro:
            messagebox.showerror("Falha na IA", str(erro))
            self._set_status("Reanálise com IA falhou.")
            return
        if not encaixes:
            self._set_status("Qwen não retornou encaixes (login pendente, janela "
                             "fechada antes da resposta ou site fora do ar); "
                             "mantendo mapeamento atual. Use '🤖 Qwen' para "
                             "verificar a sessão do navegador.")
            return
        self._encaixes_brutos = encaixes
        qwen_ok = sum(1 for e in encaixes if e.confianca >= 0.75)
        self._set_status(f"🤖 Qwen propôs {len(encaixes)} encaixe(s) "
                         f"({qwen_ok} com confiança ≥ 75%). Revisão recomendada "
                         "antes de executar.")
        self._reatualizar_mapeamentos()

    def _dialogo_ia(self) -> None:
        """
        Janela de configuração do Qwen — MODO PADRÃO: CONTA LOGADA NO NAVEGADOR.

        Não há campo de API Key obrigatório: o programa automatiza o site
        chat.qwen.ai com Playwright usando um perfil persistente do Chromium
        (~/.autodoc_qwen_profile). O fluxo humano é:
          1. Clique em "🌐 Abrir navegador e conectar minha conta".
          2. Faça login no Qwen na janela que abrir (e-mail/QR/Google).
          3. Feche a janela de teste e clique em "Salvar e testar" — a sessão
             fica gravada e toda análise futura usa sua conta, sem API.
        O modo local (Ollama) continua disponível como opção avançada para
        quem preferir rodar o modelo na própria máquina.
        """
        import qwen_navegador as qn

        win = tk.Toplevel(self.root)
        win.title("🤖 Qwen — conexão pela conta do navegador (sem API)")
        win.configure(bg=COR_PAINEL)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Como o AutoDoc fala com o Qwen", style="Titulo.TLabel"
                  ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 2))
        ttk.Label(
            win, wraplength=560, justify="left", background=COR_PAINEL,
            foreground="#374151",
            text=("NENHUMA chave de API é usada. O programa abre o chat.qwen.ai "
                  "em um navegador dedicado e conversa com o Qwen aproveitando a "
                  "SUA CONTA JÁ LOGADA. O login é feito uma única vez: a sessão "
                  "fica salva no perfil ~/.autodoc_qwen_profile e é reaproveitada "
                  "em todas as análises.")
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 8))

        # --- Bloco principal: navegador / conta logada -------------------
        grp_nav = ttk.LabelFrame(win, text=" 🌐 Modo recomendado: conta logada no navegador ")
        grp_nav.grid(row=2, column=0, columnspan=2, sticky="we", padx=14, pady=6)
        ttk.Label(grp_nav,
                  text=f"URL do chat: {qn.URL_CHAT}").pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(grp_nav,
                  text=f"Perfil do navegador: {qn.PERFIL_DIR}").pack(anchor="w", padx=10)

        lbl_res = ttk.Label(win, text="", wraplength=560,
                            foreground="#374151", justify="left")
        lbl_res.grid(row=5, column=0, columnspan=2, sticky="we", padx=14, pady=6)

        def definir_navegador() -> None:
            """Grava o modo navegador como backend ativo."""
            ConfigIAQwen(estilo="navegador").salvar()
            reset_cache_ia()
            self.cfg_ia = ConfigIAQwen.carregar()
            self._atualizar_label_ia()
            lbl_res.config(text="✔ Modo 'conta do navegador' ativado.",
                           foreground=COR_OK)

        def abrir_login() -> None:
            """Abre o chat do Qwen na janela persistente p/ login manual."""
            lbl_res.config(text="Abrindo o navegador… faça login do Qwen nessa "
                                "janela; depois volte aqui e clique em "
                                "'Salvar e testar'.", foreground="#374151")

            def bg() -> None:
                try:
                    motor = qn.MotorQwenNavegador(qn.ConfigQwenNavegador.carregar())
                    motor.abrir_chat(novo_chat=False)
                    ok = motor.esta_logado()
                    if not ok:
                        # mantém a janela aberta p/ o usuário logar
                        msg = ("Janela do Qwen aberta. Faça LOGIN nela; quando o "
                               "chat aparecer, feche esta caixa de teste e clique "
                               "em 'Salvar e testar'.")
                        self.root.after(0, lambda: lbl_res.config(
                            text="⚠ " + msg, foreground=COR_WARN))
                        return
                    definir_navegador()
                    self.root.after(0, lambda: (lbl_res.config(
                        text="✔ Login encontrado! A conta do navegador será usada "
                             "para analisar os documentos.", foreground=COR_OK),
                        messagebox.showinfo(
                            "Qwen conectado",
                            "Sua conta logada no navegador foi detectada.\n"
                            "As análises passarão por ela — sem API.")))
                except Exception as exc:  # noqa: BLE001 - fronteira de robustez
                    self.root.after(0, lambda: lbl_res.config(
                        text=f"✖ {exc}", foreground=COR_ERRO))

            threading.Thread(target=bg, daemon=True).start()

        botoes_nav = ttk.Frame(grp_nav)
        botoes_nav.pack(fill="x", padx=10, pady=8)
        ttk.Button(botoes_nav, text="🌐 Abrir navegador e conectar minha conta",
                   style="Primario.TButton", command=abrir_login).pack(side="left")
        ttk.Button(botoes_nav, text="Usar este modo",
                   command=definir_navegador).pack(side="right")

        # --- Bloco opcional: modelo local (avançado) ----------------------
        grp_local = ttk.LabelFrame(win, text=" ⚙ Modo avançado (opcional): Ollama local — não usado por padrão ")
        grp_local.grid(row=3, column=0, columnspan=2, sticky="we", padx=14, pady=6)
        ttk.Label(grp_local, text="Base URL:").grid(row=0, column=0, sticky="w", padx=10, pady=4)
        ent_url = ttk.Entry(grp_local, width=40)
        ent_url.insert(0, "http://localhost:11434")
        ent_url.grid(row=0, column=1, sticky="we", padx=(0, 10), pady=4)
        ttk.Label(grp_local, text="Modelo:").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        ent_modelo = ttk.Entry(grp_local, width=40)
        ent_modelo.insert(0, "qwen2.5:7b-instruct")
        ent_modelo.grid(row=1, column=1, sticky="we", padx=(0, 10), pady=4)
        grp_local.columnconfigure(1, weight=1)

        def definir_local() -> None:
            """Ativa o modo local apenas se o servidor estiver acessível."""
            url = ent_url.get().strip().rstrip("/")
            modelo = ent_modelo.get().strip()
            if not url:
                lbl_res.config(text="Informe uma Base URL para o modo local.",
                               foreground=COR_WARN)
                return
            cfg = ConfigIAQwen(estilo="local", base_url=url, api_key="",
                               modelo=modelo or "qwen2.5:7b-instruct")
            detectado = ConfigIAQwen._detectar_local({"base_url": url,
                                                      "modelo": modelo})
            if detectado is None:
                lbl_res.config(
                    text=f"⚠ Nenhum servidor de IA local respondendo em {url}. "
                         "Mantenha o modo navegador (recomendado).",
                    foreground=COR_WARN)
                return
            cfg.estilo = "local"
            cfg.salvar()
            reset_cache_ia()
            self.cfg_ia = ConfigIAQwen.carregar()
            self._atualizar_label_ia()
            lbl_res.config(text=f"✔ Backend local salvo ({detectado.modelo}).",
                           foreground=COR_OK)

        botoes_loc = ttk.Frame(grp_local)
        botoes_loc.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(2, 8))
        ttk.Button(botoes_loc, text="Usar Ollama local",
                   command=definir_local).pack(side="right", padx=(6, 0))

        # --- Desligar IA ---------------------------------------------------
        grp_off = ttk.Frame(win)
        grp_off.grid(row=4, column=0, columnspan=2, sticky="ew", padx=14)
        chk_desligar = ttk.Checkbutton(
            grp_off, text="Desligar a IA (usar só extração estrutural/heurística)")
        chk_desligar.pack(side="left")

        def salvar_fechando() -> None:
            if chk_desligar.get():
                ConfigIAQwen(estilo="nenhum").salvar()
            elif self.cfg_ia and self.cfg_ia.estilo == "navegador":
                ConfigIAQwen(estilo="navegador").salvar()
            reset_cache_ia()
            self.cfg_ia = ConfigIAQwen.carregar()
            self._atualizar_label_ia()
            win.destroy()

        botoes = ttk.Frame(win)
        botoes.grid(row=6, column=0, columnspan=2, sticky="ew", padx=14, pady=(4, 12))
        ttk.Button(botoes, text="💾 Salvar e fechar", style="Primario.TButton",
                   command=salvar_fechando).pack(side="right", padx=(6, 0))
        ttk.Button(botoes, text="Fechar", command=win.destroy).pack(side="right")
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)

        # Estado atual já visível ao abrir
        cfg = ConfigIAQwen.carregar()
        if cfg is None:
            lbl_res.config(text="IA desligada atualmente.", foreground=COR_WARN)
        elif cfg.estilo == "navegador":
            pronto = qn.disponivel()
            lbl_res.config(
                text=("✔ Modo navegador ativo" +
                      (" — sessão de login já encontrada!" if pronto else
                       ". Clique em 'Abrir navegador e conectar minha conta' "
                        "para fazer o primeiro login.")),
                foreground=COR_OK if pronto else "#374151")
        else:
            lbl_res.config(text=f"Backend local ativo: {cfg.base_url}",
                          foreground=COR_OK)

    def _atualizar_label_ia(self) -> None:
        """Mostra no topo o estado real do backend Qwen."""
        import qwen_navegador as qn
        cfg = self.cfg_ia
        if cfg is None:
            self.lbl_ia.config(
                text="🤖 Qwen: desligado (heurística local + fuzzy)",
                foreground=COR_WARN)
        elif cfg.estilo == "navegador":
            sessao = "sessão salva ✔" if qn.disponivel() else "clique em 🤖 Qwen p/ login"
            self.lbl_ia.config(
                text=f"🤖 Qwen: conta logada no navegador ({sessao})",
                foreground=COR_OK if qn.disponivel() else COR_WARN)
        else:
            self.lbl_ia.config(
                text=f"🤖 Qwen: {cfg.modelo} @ {cfg.base_url} (local/{cfg.estilo})",
                foreground=COR_OK)

    def _remapear_manual(self, m: MapeamentoCampo, chave: str,
                         var_valor: tk.StringVar) -> None:
        """Dropdown escolheu outro dado da origem -> override manual (100%)."""
        if not self.resultado:
            return
        if chave == "— nenhum —":
            m.dado, m.valor_final, m.score = None, "", 0.0
            m.status = StatusCampo.SEM_DADO
            m.justificativa = "limpo manualmente pelo usuário"
        else:
            d = next((x for x in self.resultado.dados
                      if x.chave_original == chave), None)
            if d:
                m.dado, m.valor_final, m.score = d, d.valor, 1.0
                m.status = StatusCampo.OK
                m.justificativa = f"atribuído manualmente a '{d.chave_original}'"
        m.origem_encaixe = "manual"
        var_valor.set(m.valor_final)
        self._reatualizar_mapeamentos()

    # --------------------------------------------------------------- execução
    def _executar_preenchimento(self, dry_run: bool) -> None:
        """Valida pré-requisitos e delega ao executor correto em thread."""
        self._sincronizar_valores_ui()
        if not self.mapas:
            messagebox.showwarning("Nada a fazer",
                                   "Carregue origem E destino antes de preencher.")
            return
        pendentes = [m for m in self.mapas if m.precisa_revisao and not m.valor_final]
        if not dry_run and pendentes:
            if not messagebox.askyesno(
                    "Campos sem revisão",
                    f"{len(pendentes)} campo(s) ainda marcados para revisão serão "
                    "pulados. Continuar assim mesmo?"):
                return
        rotulo = "SIMULAÇÃO" if dry_run else "EXECUÇÃO"
        self._set_status(f"[{rotulo}] processando ...")
        threading.Thread(target=self._run_fill_bg, args=(dry_run,),
                         daemon=True).start()

    def _run_fill_bg(self, dry_run: bool) -> None:
        """Chama o executor adequado (web/pdf/txt) fora da main thread."""
        try:
            if self.destino_url:
                mapas = PreenchedorWeb().preencher(self.destino_url, self.mapas, dry_run)
                msg = (f"[{dry_run and 'SIMULAÇÃO' or 'EXECUÇÃO'}] web concluída — "
                       f"resultado_web.png salvo.")
            elif self.destino_arquivo and self.destino_arquivo.suffix.lower() == ".pdf":
                saida = self.destino_arquivo.with_name(
                    ("dryrun_" if dry_run else "") + "saida_" + self.destino_arquivo.name)
                out = PreenchedorPDF(self.destino_arquivo).preencher(
                    self.mapas, saida, dry_run)
                mapas, msg = self.mapas, f"PDF gerado: {out}"
            elif self.destino_arquivo:
                saida = self.destino_arquivo.with_name(
                    ("dryrun_" if dry_run else "") + "saida_" + self.destino_arquivo.name)
                out = PreenchedorTXT(self.destino_arquivo).preencher(
                    self.mapas, saida, dry_run)
                mapas, msg = self.mapas, f"TXT gerado: {out}"
            else:
                mapas, msg = self.mapas, "Nenhum destino configurado."
            self.root.after(0, self._on_fill_done, mapas, msg, None)
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._on_fill_done, self.mapas, "", exc)

    def _on_fill_done(self, mapas: list[MapeamentoCampo], msg: str,
                      erro: Optional[Exception]) -> None:
        """Atualiza painel direito com status final (verde/laranja)."""
        self.mapas = mapas
        if erro:
            messagebox.showerror("Falha no preenchimento", str(erro))
            self._set_status(f"Erro: {erro}")
        else:
            self._set_status(msg)
        self._reatualizar_mapeamentos()
        ok = sum(1 for m in mapas if m.status in (StatusCampo.PREENCHIDO, StatusCampo.OK))
        falha = sum(1 for m in mapas if m.status in (StatusCampo.FALHA, StatusCampo.REVISAO))
        logger.info("Fill finalizado: %d ok / %d pendentes", ok, falha)

    # ---------------------------------------------------------------- helpers
    def _set_status(self, texto: str) -> None:
        """Atualiza a barra de status inferior."""
        self.status_var.set(texto)


# ---------------------------------------------------------------------------
# Wrappers de simpledialog (import tardio mantém módulo leve p/ testes headless)
# ---------------------------------------------------------------------------
def simpledialog_get_password(parent: Any, title: str, prompt: str) -> Optional[str]:
    """Pede senha de PDF criptografado."""
    from tkinter import simpledialog
    return simpledialog.askstring(title, prompt, show="*", parent=parent)


def simpledialog_askstring(parent: Any, title: str, initial: str) -> Optional[str]:
    """Pede uma URL de formulário web."""
    from tkinter import simpledialog
    return simpledialog.askstring(title, "URL do formulário:",
                                  initialvalue=initial, parent=parent)


def main() -> None:
    """Ponto de entrada da aplicação."""
    root = tk.Tk()
    AplicacaoPreenchimento(root)
    root.mainloop()


if __name__ == "__main__":
    main()
