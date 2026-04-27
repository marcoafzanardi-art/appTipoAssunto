"""
Motor de Classificação de Documentos
Streamlit app para classificação automática de documentos arquivísticos.
"""

import streamlit as st
import pandas as pd
import io
from classifier import classify_dataframe

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Motor de Classificação de Documentos",
    page_icon="📂",
    layout="wide",
)

# ─── CSS ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0f0f0f;
    --surface: #1a1a1a;
    --surface2: #242424;
    --border: #2e2e2e;
    --accent: #e8c547;
    --accent2: #4ecdc4;
    --text: #f0ece0;
    --text-muted: #888;
    --red: #ff6b6b;
    --green: #6bcb77;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Header */
.hero {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: var(--accent);
    margin: 0;
    letter-spacing: -1px;
    line-height: 1;
}
.hero p {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
}

/* Upload box */
.upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    background: var(--surface);
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* Stats cards */
.stats-grid { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    min-width: 140px;
    flex: 1;
}
.stat-card .value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent);
    line-height: 1;
}
.stat-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Confidence badge */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-high { background: rgba(107,203,119,0.15); color: var(--green); }
.badge-mid  { background: rgba(232,197,71,0.15);  color: var(--accent); }
.badge-low  { background: rgba(255,107,107,0.15); color: var(--red); }

/* Streamlit overrides */
.stFileUploader > div { background: var(--surface) !important; border: 1.5px dashed var(--border) !important; border-radius: 8px !important; }
.stFileUploader label { color: var(--text) !important; }
div[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; }
div[data-testid="stFileUploaderDropzone"] p { color: var(--text-muted) !important; }

.stButton > button {
    background: var(--accent) !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stDownloadButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stDownloadButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden; }
div[data-testid="stDataFrame"] { background: var(--surface) !important; }

.stSelectbox > div > div { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }

section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stProgress > div > div > div { background: var(--accent) !important; }

h2, h3 { font-family: 'DM Serif Display', serif; color: var(--text); }
.stMarkdown a { color: var(--accent2) !important; }

hr { border-color: var(--border) !important; }

/* Info / warning */
div[data-testid="stAlert"] { border-radius: 8px !important; border-left: 3px solid var(--accent) !important; background: var(--surface) !important; }

/* Expander */
details { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
summary { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>📂 Motor de Classificação</h1>
    <p>Classificação automática de documentos arquivísticos</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    st.markdown("---")
    st.markdown("**Colunas esperadas no arquivo:**")
    st.markdown("- `DOC_ID` *(obrigatório)*")
    st.markdown("- `TIPO_ANTIGO` *(obrigatório)*")
    st.markdown("- `OBJETO` *(opcional)*")
    st.markdown("- `CONTEUDO` *(opcional)*")
    st.markdown("---")
    st.markdown("**Níveis de confiança:**")
    st.markdown("🟢 **≥ 0.95** — Automático")
    st.markdown("🟡 **0.80–0.94** — Revisão rápida")
    st.markdown("🟠 **0.60–0.79** — Revisão obrigatória")
    st.markdown("🔴 **< 0.60** — Classificação manual")
    st.markdown("---")
    st.markdown("**Métodos:**")
    st.markdown("- `regra` — Regra de negócio exata")
    st.markdown("- `similaridade` — TF-IDF + cosseno")

# ─── Main ────────────────────────────────────────────────────────────────────

st.markdown("### 1 · Carregar arquivo")

uploaded = st.file_uploader(
    "Selecione um arquivo CSV ou Excel",
    type=["csv", "xlsx", "xls"],
    help="O arquivo deve conter as colunas DOC_ID, TIPO_ANTIGO e opcionalmente OBJETO e CONTEUDO.",
)

if uploaded is not None:
    # ── Load ────────────────────────────────────────────────────────────────
    try:
        if uploaded.name.endswith(".csv"):
            # Try semicolon first, then comma
            try:
                df_in = pd.read_csv(uploaded, sep=";", encoding="utf-8", dtype=str)
                if df_in.shape[1] == 1:
                    uploaded.seek(0)
                    df_in = pd.read_csv(uploaded, sep=",", encoding="utf-8", dtype=str)
            except Exception:
                uploaded.seek(0)
                df_in = pd.read_csv(uploaded, sep=",", encoding="latin-1", dtype=str)
        else:
            df_in = pd.read_excel(uploaded, dtype=str)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # Normalize column names
    df_in.columns = [c.strip().upper() for c in df_in.columns]

    # Validate required columns
    missing = [c for c in ["DOC_ID", "TIPO_ANTIGO"] if c not in df_in.columns]
    if missing:
        st.error(f"Colunas obrigatórias ausentes: {', '.join(missing)}")
        st.stop()

    # Fill optional columns
    for col in ["OBJETO", "CONTEUDO"]:
        if col not in df_in.columns:
            df_in[col] = ""

    df_in = df_in.fillna("")

    st.success(f"✅ Arquivo carregado com **{len(df_in):,}** linhas e **{len(df_in.columns)}** colunas.")

    with st.expander("👁 Pré-visualização do arquivo carregado"):
        st.dataframe(df_in.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2 · Classificar")

    if st.button("🚀 Iniciar Classificação", use_container_width=False):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(pct: float, msg: str):
            progress_bar.progress(pct)
            status_text.markdown(f"<span style='font-family:DM Mono,monospace;font-size:0.85rem;color:#888'>{msg}</span>", unsafe_allow_html=True)

        with st.spinner("Processando…"):
            df_out = classify_dataframe(df_in.copy(), progress_callback=progress_callback)

        progress_bar.progress(1.0)
        status_text.markdown("<span style='color:#6bcb77;font-family:DM Mono,monospace;font-size:0.85rem'>✓ Classificação concluída!</span>", unsafe_allow_html=True)

        st.session_state["df_out"] = df_out

# ─── Results ────────────────────────────────────────────────────────────────

if "df_out" in st.session_state:
    df_out = st.session_state["df_out"]

    st.markdown("---")
    st.markdown("### 3 · Resultados")

    # Stats
    total = len(df_out)
    auto    = (df_out["CONFIANCA"] >= 0.95).sum()
    review  = ((df_out["CONFIANCA"] >= 0.60) & (df_out["CONFIANCA"] < 0.95)).sum()
    manual  = (df_out["CONFIANCA"] < 0.60).sum()
    regra   = (df_out["METODO"] == "regra").sum()

    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-card"><div class="value">{total:,}</div><div class="label">Total</div></div>
      <div class="stat-card"><div class="value" style="color:#6bcb77">{auto:,}</div><div class="label">Automático ≥95%</div></div>
      <div class="stat-card"><div class="value" style="color:#e8c547">{review:,}</div><div class="label">Revisão 60–94%</div></div>
      <div class="stat-card"><div class="value" style="color:#ff6b6b">{manual:,}</div><div class="label">Manual &lt;60%</div></div>
      <div class="stat-card"><div class="value" style="color:#4ecdc4">{regra:,}</div><div class="label">Por regra</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Filter
    col_filter, col_method = st.columns(2)
    with col_filter:
        confianca_filter = st.selectbox(
            "Filtrar por nível de confiança",
            ["Todos", "Automático (≥0.95)", "Revisão rápida (0.80–0.94)", "Revisão obrigatória (0.60–0.79)", "Manual (<0.60)"]
        )
    with col_method:
        method_filter = st.selectbox("Filtrar por método", ["Todos", "regra", "similaridade"])

    df_view = df_out.copy()
    if confianca_filter == "Automático (≥0.95)":
        df_view = df_view[df_view["CONFIANCA"] >= 0.95]
    elif confianca_filter == "Revisão rápida (0.80–0.94)":
        df_view = df_view[(df_view["CONFIANCA"] >= 0.80) & (df_view["CONFIANCA"] < 0.95)]
    elif confianca_filter == "Revisão obrigatória (0.60–0.79)":
        df_view = df_view[(df_view["CONFIANCA"] >= 0.60) & (df_view["CONFIANCA"] < 0.80)]
    elif confianca_filter == "Manual (<0.60)":
        df_view = df_view[df_view["CONFIANCA"] < 0.60]

    if method_filter != "Todos":
        df_view = df_view[df_view["METODO"] == method_filter]

    st.markdown(f"*Exibindo {len(df_view):,} de {total:,} linhas*")

    # Display key columns prominently
    display_cols = ["DOC_ID", "TIPO_ANTIGO", "OBJETO", "TIPO_CANONICO", "CONFIANCA", "METODO", "REVISAO_NECESSARIA"]
    display_cols = [c for c in display_cols if c in df_view.columns]

    def color_confianca(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v >= 0.95:
            return "background-color: #d1fae5; color: #065f46"
        elif v >= 0.80:
            return "background-color: #fef9c3; color: #713f12"
        elif v >= 0.60:
            return "background-color: #ffedd5; color: #7c2d12"
        else:
            return "background-color: #fee2e2; color: #7f1d1d"

    st.dataframe(
        df_view[display_cols].style
            .format({"CONFIANCA": "{:.2f}"})
            .map(color_confianca, subset=["CONFIANCA"]),
        use_container_width=True,
        height=420,
    )

    # Distribution chart
    with st.expander("📊 Distribuição de tipos canônicos"):
        top = df_out["TIPO_CANONICO"].value_counts().head(20)
        st.bar_chart(top)

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4 · Download")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        excel_buffer = io.BytesIO()
        df_out.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        st.download_button(
            label="⬇ Baixar Excel (.xlsx)",
            data=excel_buffer,
            file_name="documentos_classificados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col_dl2:
        csv_data = df_out.to_csv(index=False, sep=";", encoding="utf-8-sig")
        st.download_button(
            label="⬇ Baixar CSV (.csv)",
            data=csv_data,
            file_name="documentos_classificados.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #555; font-family: 'DM Mono', monospace; font-size:0.9rem;">
            ↑ Faça o upload de um arquivo CSV ou Excel para começar.
        </div>
        """, unsafe_allow_html=True)
