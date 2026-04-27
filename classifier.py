"""
classifier.py — Motor de Classificação de Documentos
Limpeza, regras de negócio e similaridade semântica (TF-IDF).
"""

import re
import unicodedata
from typing import Optional, Callable
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# 1. TIPOS CANÔNICOS
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_TYPES = [
    "ACORDO COLETIVO",
    "APOLICE DE SEGURO",
    "ATA DE REUNIAO",
    "ATENDIMENTO MEDICO",
    "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO",
    "ATESTADO MEDICO",
    "AUDITORIA DE CONTROLE INTERNO",
    "AUDITORIA DE CONTROLE INTERNO - DEMONSTRACOES CONTABEIS",
    "AUDITORIA DE CONTROLE INTERNO - LICITACOES E CONTRATOS",
    "AUTORIZACAO DE DESCONTO EM FOLHA - ASSISTENCIA MEDICA",
    "AUTORIZACAO DE VIAGEM",
    "AUXILIO DOENCA - INSS",
    "BRIEFING DE CAMPANHA",
    "CALENDARIO",
    "COMUNICACAO EXTERNA",
    "COMUNICACAO INTERNA",
    "CONTRATO",
    "CONTRATO DE CESSAO DE USO",
    "CONTRATO DE CESSAO TEMPORARIA DE USO",
    "CONTRATO DE CONCESSAO DE USO DE IMAGEM E VOZ",
    "CONTROLE DE ACESSO DE VISITANTE",
    "CONTROLE DE FREQUENCIA E OCORRENCIAS DE PESSOAL",
    "CONTROLE DE PUBLICACOES",
    "CORRESPONDENCIA EXTERNA",
    "CORRESPONDENCIA INTERNA",
    "DECLARACAO DE BENS",
    "DEMONSTRATIVO CONTABIL",
    "DEMONSTRATIVO DE FATURAMENTO",
    "DOSSIE DE EVENTO",
    "DOSSIE DE PUBLICIDADE",
    "FICHA FINANCEIRA",
    "FOLHA DE FREQUENCIA",
    "FOLHA DE PAGAMENTO",
    "FOLHETO EXEMPLAR",
    "GESTAO E FISCALIZACAO DE CONTRATO - AUTORIZACAO DE SERVICO",
    "GESTAO E FISCALIZACAO DE CONTRATO - CONTROLE DE PESSOAL TERCEIRIZADO",
    "GESTAO E FISCALIZACAO DE CONTRATO - CORRESPONDENCIA INTERNA",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO ADMINISTRATIVA E COMUNICACOES",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO ADMINISTRATIVA E CONTRATUAL",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO DE SEGURANCA E SAUDE DO TRABALHO",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO FINANCEIRA E DE PAGAMENTOS",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO OPERACIONAL E DE EXECUCAO DOS SERVICOS",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO TRABALHISTA E PREVIDENCIARIA DE TERCEIRIZADOS",
    "GESTAO E FISCALIZACAO DE CONTRATO - FICHA DE REGISTRO DE TERCEIRIZADO",
    "GESTAO E FISCALIZACAO DE CONTRATO - FOLHA DE FREQUENCIA",
    "GESTAO E FISCALIZACAO DE CONTRATO - FOLHA DE PAGAMENTO DE TERCEIRIZADOS",
    "GESTAO E FISCALIZACAO DE CONTRATO - GUIA DE RECOLHIMENTO DE FGTS",
    "GUIA DE RECOLHIMENTO DE INSS",
    "INTIMACAO",
    "JORNAL EXEMPLAR",
    "LIVRO",
    "LIVRO DE APURACAO DE LUCRO REAL",
    "LIVRO EXEMPLAR",
    "LIVRO RAZAO",
    "MATERIAL PROMOCIONAL",
    "MIDIA FISICA EXEMPLAR",
    "OFICIO",
    "PAGAMENTO DE SERVICOS - ORDEM BANCARIA",
    "PARECER JURIDICO",
    "PRESTACAO DE CONTAS - CONVENIO",
    "PROCESSO ADMINISTRATIVO - ADIANTAMENTO",
    "PROCESSO ADMINISTRATIVO - ALIENACAO DE BENS",
    "PROCESSO ADMINISTRATIVO - CONTRATACAO",
    "PROCESSO ADMINISTRATIVO - GESTAO E FISCALIZACAO DE CONTRATO",
    "PROCESSO ADMINISTRATIVO - PAGAMENTO",
    "PROCESSO ADMINISTRATIVO - PROVIMENTO DE CARGO",
    "PROCESSO CIVEL E TRABALHISTA",
    "PROCESSO DE AUDITORIA EM CONTRATACOES",
    "PROCESSO DE COMPRA",
    "PROCESSO DE EVENTO",
    "PROCESSO DE PAGAMENTO DE TRIBUTOS",
    "PROCESSO DE PAGAMENTO DE TRIBUTOS - DARF",
    "PROCESSO JUDICIAL",
    "PROCESSO JUDICIAL - ACAO DE COBRANCA",
    "PRONTUARIO DE ESTAGIARIO",
    "PRONTUARIO DO SERVIDOR",
    "PRONTUARIO MEDICO",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO - ADMISSAO DE PESSOAL",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO - DEMONSTRACOES CONTABEIS",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO - LICITACOES E CONTRATOS",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO - QUADRO DE PESSOAL",
    "RELATORIO DE RETENCOES",
    "REVISTA EXEMPLAR",
    "TERMO DE POSSE",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DICIONÁRIO ARQUIVÍSTICO (contexto para similaridade)
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVAL_DICT: dict[str, str] = {
    "PROCESSO JUDICIAL": "acao cobranca mandado de seguranca execucao fiscal trabalhista civel indenizacao inquerito policial notificacao extrajudicial TCM alvara jurisprudencia procurador liminar tutela de urgencia rito ordinario embargos de terceiros acordo judicial",
    "PROCESSO JUDICIAL - ACAO DE COBRANCA": "cobranca rito ordinario volume",
    "PROCESSO DE COMPRA": "licitacao pregao edital fornecedor contratacao locacao aquisicao concorrencia proposta carregadores ventiladores engenheiro lixeira seletiva",
    "PROCESSO ADMINISTRATIVO - CONTRATACAO": "chamamento publico licitacao edital concorrencia internacional seguro veiculo",
    "PROCESSO ADMINISTRATIVO - GESTAO E FISCALIZACAO DE CONTRATO": "contrato termo de contrato ordem de inicio de servico memorial descritivo gestao fiscalizacao dossie",
    "PAGAMENTO DE SERVICOS - ORDEM BANCARIA": "pagamento ordem bancaria fatura nota fiscal despesa bloco de pagamento expediente de pagamentos",
    "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO": "AELP liberacao execucao pagamento medicao servico prestado GCP GTD SGC GEM GIE autodromo",
    "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO ADMINISTRATIVA E CONTRATUAL": "contrato terceirizado folha de pagamento frequencia PPRA PCMSO laudo tecnico diario de obra autorizacao de servico protocolo de entrega controle de pessoal",
    "AUDITORIA DE CONTROLE INTERNO": "TCM licitacao contrato admissao demissao contas anuais irregularidades papeis de trabalho achados de auditoria",
    "RELATORIO DE AUDITORIA DE CONTROLE INTERNO": "auditoria licitacoes contratos demonstracoes contabeis admissao de pessoal quadro de pessoal",
    "DOSSIE DE EVENTO": "carnaval Indy 300 memorial descritivo apresentacao sorteio autorizacao de uso de espaco",
    "DOSSIE DE PUBLICIDADE": "campanha publicitaria midia fisica jornal folheto exemplar anuncio controle de veiculacao revista apostila cartao fisico",
    "PRONTUARIO DO SERVIDOR": "copia conselheiro funcional prontuario servidor",
    "ATENDIMENTO MEDICO": "mapa de atendimento avaliacao medica campanha ex-funcionario afastado",
    "FOLHA DE FREQUENCIA": "ponto frequencia folha de ponto individual",
    "FOLHA DE PAGAMENTO": "pagamento autonomos lancamentos controle de pagamento",
    "PROCESSO DE PAGAMENTO DE TRIBUTOS": "DARF INSS FGTS guia de recolhimento memoria de calculo",
    "OFICIO": "TCM CVM CHG CIPA CRE GCP AP GAI GJU ministerio publico delegacia",
    "CORRESPONDENCIA INTERNA": "comunicacao interna e-mail CI DAF GJU ouvidoria memorando",
    "CORRESPONDENCIA EXTERNA": "oficio externo comunicacao externa declaracao",
    "CONTRATO": "prestacao de servicos locacao cessao termo contrato administrativo aditivo",
    "ATA DE REUNIAO": "pauta reuniao COMTUR assembleia diretoria executiva deliberacao",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIMPEZA E NORMALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

# Encoding artifacts to fix before unicode normalization
ENCODING_FIXES = [
    ("Ã‡", "C"), ("Ãƒ", "A"), ("Ã", "A"), ("Â", " "),
    ("â€œ", '"'), ("â€", '"'), ("&quot", '"'), ("&apos", "'"),
]

# Abbreviations to expand (NOT domain siglas)
ABBREVIATIONS = {
    r"\bRH\b": "RECURSOS HUMANOS",
    r"\bPREST SERV\b": "PRESTACAO DE SERVICOS",
    r"\bPREST\. SERV\b": "PRESTACAO DE SERVICOS",
}

# Domain siglas to preserve (never expand)
PRESERVE_SIGLAS = {
    "FGTS", "INSS", "DARF", "PPRA", "PCMSO", "TCM", "AVCB", "AELP",
    "CIPA", "GJU", "DAF", "CHG", "GCP", "GTD", "GEM", "GIE", "SGC",
}


def _remove_accents(text: str) -> str:
    """Remove diacritics using unicodedata (no external dependency for this step)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(raw: str) -> str:
    """
    Full cleaning pipeline:
    1. Fix encoding artefacts
    2. Uppercase
    3. Remove accents
    4. Expand abbreviations
    5. Collapse whitespace
    """
    if not isinstance(raw, str):
        return ""

    text = raw
    # Fix encoding artefacts
    for bad, good in ENCODING_FIXES:
        text = text.replace(bad, good)

    # Uppercase
    text = text.upper()

    # Remove accents
    text = _remove_accents(text)

    # Expand abbreviations
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    # Collapse whitespace / tabs / newlines
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4. REGRAS DE CLASSIFICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _contains(text: str, substring: str) -> bool:
    return substring in text


def _is_number_code(text: str) -> bool:
    """Check if TIPO_ANTIGO looks like a numeric code (e.g. 0187/14, 1458/2014)."""
    return bool(re.match(r"^\d+[/\-]\d+$", text.strip()))


def apply_rules(tipo: str, objeto: str) -> Optional[str]:
    """
    Apply business rules in priority order.
    Returns canonical type string or None if no rule matched.
    """

    # ── PAGAMENTO ──────────────────────────────────────────────────────────
    if _contains(tipo, "BORDERO DE PAGAMENTOS"):
        return "PAGAMENTO DE SERVICOS - ORDEM BANCARIA"
    if _contains(tipo, "DOSSIE") and any(_contains(tipo, x) for x in
                                          ["PAGAMENTO", "PAGAMNTO", "PAGEMNTO", "PAGAMENTTO", "PGAMENTO"]):
        return "PAGAMENTO DE SERVICOS - ORDEM BANCARIA"
    if tipo == "EXPEDIENTE" and _contains(objeto, "PAGAMENTOS DIVERSOS"):
        return "PAGAMENTO DE SERVICOS - ORDEM BANCARIA"
    if tipo == "EXPEDICAO" and _contains(objeto, "PAGAMENTOS"):
        return "PAGAMENTO DE SERVICOS - ORDEM BANCARIA"
    if tipo == "EXPEDIENTE DE PAGAMENTO":
        return "PAGAMENTO DE SERVICOS - ORDEM BANCARIA"

    # ── ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO ─────────────────────
    if _contains(tipo, "AELP"):
        return "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO"
    if _contains(tipo, "ATESTADO DE EXECUCAO E LIBERACAO"):
        return "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO"
    if _contains(tipo, "LIBERACAO DE PAGAMENTO"):
        return "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO"
    if _contains(objeto, "AELP"):
        return "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO"
    if _contains(tipo, "ATESTADO DE LIBERACAO E EXECUCAO"):
        return "ATESTADO DE EXECUCAO E LIBERACAO DE PAGAMENTO"

    # ── PROCESSO DE COMPRA ─────────────────────────────────────────────────
    if any(_contains(tipo, x) for x in ["COMPRA", "COMPRAS", "COMRAS", "COMPAS"]):
        return "PROCESSO DE COMPRA"
    if tipo == "PROCEDIMENTO ADMINISTRATIVO" and _contains(objeto, "LOCACAO"):
        return "PROCESSO DE COMPRA"
    if tipo == "SOLICITACAO DE COMPRAS":
        return "PROCESSO DE COMPRA"
    if _is_number_code(tipo) and _contains(objeto, "CONTRATACAO"):
        return "PROCESSO DE COMPRA"

    # ── PROCESSO JUDICIAL ─ ACAO DE COBRANCA (antes do genérico) ──────────
    if _contains(tipo, "ACAO DE COBRANCA"):
        return "PROCESSO JUDICIAL - ACAO DE COBRANCA"

    # ── PROCESSO JUDICIAL ─────────────────────────────────────────────────
    if _contains(tipo, "ACAO") and not _contains(tipo, "COBRANCA"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "AGRAVO DE INSTRUMENTO"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "EXECUCAO FISCAL"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "MANDADO DE SEGURANCA"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "INQUERITO POLICIAL"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "RECLAMACAO TRABALHISTA"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "NOTIFICACAO JUDICIAL") or _contains(tipo, "NOTIFICACAO EXTRAJUDICIAL"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "EMBARGOS"):
        return "PROCESSO JUDICIAL"
    if _contains(tipo, "ALVARA") and not _contains(tipo, "SOLICITACAO"):
        return "PROCESSO JUDICIAL"
    if tipo == "DOSSIE" and _contains(objeto, "TCM"):
        return "PROCESSO JUDICIAL"
    if _contains(objeto, "PROCESSO TRABALHISTA") or _contains(objeto, "ACAO TRABALHISTA"):
        return "PROCESSO JUDICIAL"
    if tipo in ("JURISPRUDENCIA", "JURISPRUDENCIA CRIMINAL"):
        return "PROCESSO JUDICIAL"
    if tipo == "PROCURACAO":
        return "PROCESSO JUDICIAL"
    if tipo == "REQUERIMENTO":
        return "PROCESSO JUDICIAL"

    # ── PROCESSO ADMINISTRATIVO ────────────────────────────────────────────
    if tipo == "PROCESSO ADMINISTRATIVO":
        if _contains(objeto, "LICITACAO"):
            return "PROCESSO ADMINISTRATIVO - CONTRATACAO"
        if _contains(objeto, "CHAMAMENTO PUBLICO"):
            return "PROCESSO ADMINISTRATIVO - CONTRATACAO"
        if _contains(objeto, "GESTAO E FISCALIZACAO"):
            return "PROCESSO ADMINISTRATIVO - GESTAO E FISCALIZACAO DE CONTRATO"
        if _contains(objeto, "ADIANTAMENTO"):
            return "PROCESSO ADMINISTRATIVO - ADIANTAMENTO"
        if _contains(objeto, "ALIENACAO"):
            return "PROCESSO ADMINISTRATIVO - ALIENACAO DE BENS"
        if _contains(objeto, "PROVIMENTO"):
            return "PROCESSO ADMINISTRATIVO - PROVIMENTO DE CARGO"
        if _contains(objeto, "PAGAMENTO"):
            return "PROCESSO ADMINISTRATIVO - PAGAMENTO"

    # ── GESTAO E FISCALIZACAO DE CONTRATO ──────────────────────────────────
    if _contains(objeto, "GESTAO E FISCALIZACAO") or _contains(tipo, "GESTAO E FISCALIZACAO"):
        return "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO ADMINISTRATIVA E CONTRATUAL"
    if tipo == "CONTRATO - EXECUCAO CONTRATUAL":
        return "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO ADMINISTRATIVA E CONTRATUAL"
    if _contains(objeto, "FATURAS"):
        return "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO FINANCEIRA E DE PAGAMENTOS"
    if _contains(objeto, "DIARIO DE OBRA"):
        return "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO OPERACIONAL E DE EXECUCAO DOS SERVICOS"
    if _contains(objeto, "PPRA") or _contains(objeto, "PCMSO") or (
            _contains(objeto, "LAUDO") and _contains(objeto, "SEGURANCA")):
        return "GESTAO E FISCALIZACAO DE CONTRATO - DOCUMENTACAO DE SEGURANCA E SAUDE DO TRABALHO"

    # ── PESSOAL ────────────────────────────────────────────────────────────
    if _contains(objeto, "PRONTUARIO") and _contains(objeto, "ESTAGIO"):
        return "PRONTUARIO DE ESTAGIARIO"
    if _contains(objeto, "PRONTUARIO") and _contains(objeto, "SERVIDOR"):
        return "PRONTUARIO DO SERVIDOR"
    if _contains(objeto, "ATENDIMENTO MEDICO"):
        return "ATENDIMENTO MEDICO"
    if tipo == "ATESTADO MEDICO":
        return "ATESTADO MEDICO"
    if tipo in ("FOLHA DE PONTO", "FOLHA DE FREQUENCIA"):
        return "FOLHA DE FREQUENCIA"
    if _contains(objeto, "FOLHA DE PAGAMENTO") and not _contains(objeto, "CONTRATO"):
        return "FOLHA DE PAGAMENTO"
    if tipo == "RECURSOS HUMANOS PONTO":
        return "CONTROLE DE FREQUENCIA E OCORRENCIAS DE PESSOAL"
    if tipo == "AUXILIO DOENCA" and _contains(objeto, "INSS"):
        return "AUXILIO DOENCA - INSS"

    # ── TRIBUTOS ───────────────────────────────────────────────────────────
    if _contains(tipo, "DARF"):
        return "PROCESSO DE PAGAMENTO DE TRIBUTOS - DARF"
    if tipo in ("FGTS - INSS", "FGTS-INSS") and _contains(objeto, "MEMORIA DE CALCULO"):
        return "PROCESSO DE PAGAMENTO DE TRIBUTOS"
    if tipo == "INSS" and _contains(objeto, "GUIA DE RECOLHIMENTO"):
        return "GUIA DE RECOLHIMENTO DE INSS"

    # ── PUBLICIDADE ────────────────────────────────────────────────────────
    if _contains(objeto, "PUBLICACAO") and _contains(objeto, "JORNAL"):
        return "JORNAL EXEMPLAR"
    if _contains(objeto, "PUBLICACAO") and _contains(objeto, "FOLHETO"):
        return "FOLHETO EXEMPLAR"
    if _contains(objeto, "MIDIA FISICA"):
        return "MIDIA FISICA EXEMPLAR"
    if _contains(tipo, "CAMPANHA PUBLICITARIA"):
        return "MATERIAL PROMOCIONAL"

    # ── OFICIO E CORRESPONDENCIA ───────────────────────────────────────────
    if _contains(tipo, "OFICIO") or _contains(tipo, "OFICIOS"):
        return "OFICIO"
    if tipo == "CORRESPONDENCIA" and _contains(objeto, "INTERNA"):
        return "CORRESPONDENCIA INTERNA"
    if tipo == "CORRESPONDENCIA" and _contains(objeto, "EXTERNA"):
        return "CORRESPONDENCIA EXTERNA"
    if tipo == "E-MAIL":
        return "CORRESPONDENCIA INTERNA"
    if tipo == "COMUNICACAO INTERNA" and not _contains(objeto, "EXTERNA"):
        return "CORRESPONDENCIA INTERNA"

    # ── EVENTO ─────────────────────────────────────────────────────────────
    if _contains(tipo, "MEMORIAL DESCRITIVO"):
        return "DOSSIE DE EVENTO"
    if _contains(objeto, "CARNAVAL") and (
            _contains(objeto, "APRESENTACAO") or _contains(objeto, "MEMORIAL")):
        return "DOSSIE DE EVENTO"
    if tipo == "APRESENTACAO" and _contains(objeto, "INDY"):
        return "DOSSIE DE EVENTO"

    # ── AUDITORIA ──────────────────────────────────────────────────────────
    if tipo == "AUDITORIA DE LICITACAO":
        return "RELATORIO DE AUDITORIA DE CONTROLE INTERNO - LICITACOES E CONTRATOS"
    if tipo == "AUDITORIA" and _contains(objeto, "ADMISSAO"):
        return "AUDITORIA DE CONTROLE INTERNO"
    if tipo == "AUDITORIA" and _contains(objeto, "LICITACAO"):
        return "AUDITORIA DE CONTROLE INTERNO - LICITACOES E CONTRATOS"

    # ── ATA DE REUNIAO ─────────────────────────────────────────────────────
    if _contains(tipo, "ATA DE REUNIAO") or tipo == "PAUTA REUNIAO":
        return "ATA DE REUNIAO"

    # ── CONTRATO ───────────────────────────────────────────────────────────
    if tipo == "TERMO DE CONTRATO":
        return "CONTRATO"
    if tipo == "CONTRATO DE LOCACAO":
        return "CONTRATO"
    if _contains(tipo, "CONTRATO DE CESSAO TEMPORARIA"):
        return "CONTRATO DE CESSAO TEMPORARIA DE USO"
    if _contains(tipo, "CONTRATO DE CESSAO DE USO"):
        return "CONTRATO DE CESSAO DE USO"
    if _contains(tipo, "CONCESSAO DE USO DE IMAGEM"):
        return "CONTRATO DE CONCESSAO DE USO DE IMAGEM E VOZ"

    return None  # No rule matched


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMILARIDADE SEMÂNTICA (TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

def _build_canonical_corpus() -> tuple[list[str], list[str]]:
    """
    Build the corpus for TF-IDF by combining canonical type names
    with their archival dictionary entries.
    """
    labels = []
    docs = []
    for ctype in CANONICAL_TYPES:
        label = ctype
        # Base: the canonical name itself
        doc_parts = [ctype.replace("-", " ")]
        # Enrich with dictionary if available
        dict_entry = ARCHIVAL_DICT.get(ctype, "")
        if dict_entry:
            doc_parts.append(dict_entry)
        labels.append(label)
        docs.append(" ".join(doc_parts))
    return labels, docs


# Build vectorizer once at module level
_canonical_labels, _canonical_docs = _build_canonical_corpus()
_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)
_canonical_matrix = _vectorizer.fit_transform(_canonical_docs)


def classify_by_similarity(tipo: str, objeto: str) -> tuple[str, float]:
    """
    Classify using TF-IDF + cosine similarity.
    Returns (canonical_type, confidence_score).
    """
    query = f"{tipo} {objeto}".strip()
    if not query:
        return "A CLASSIFICAR", 0.0

    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _canonical_matrix).flatten()
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    if best_score < 0.60:
        return "A CLASSIFICAR", best_score
    return _canonical_labels[best_idx], best_score


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN CLASSIFICATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def classify_row(tipo_raw: str, objeto_raw: str) -> dict:
    """
    Full pipeline for a single row.
    Returns dict with TIPO_CANONICO, CONFIANCA, METODO, REVISAO_NECESSARIA.
    """
    tipo = clean_text(tipo_raw)
    objeto = clean_text(objeto_raw)

    # Step 1: try rules
    matched = apply_rules(tipo, objeto)
    if matched:
        return {
            "TIPO_CANONICO": matched,
            "CONFIANCA": 1.0,
            "METODO": "regra",
            "REVISAO_NECESSARIA": False,
        }

    # Step 2: similarity
    best_type, score = classify_by_similarity(tipo, objeto)
    return {
        "TIPO_CANONICO": best_type,
        "CONFIANCA": round(score, 4),
        "METODO": "similaridade",
        "REVISAO_NECESSARIA": score < 0.95,
    }


def classify_dataframe(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Classify an entire DataFrame.
    Adds columns: TIPO_CANONICO, CONFIANCA, METODO, REVISAO_NECESSARIA.
    Preserves all original columns.
    """
    results = []
    total = len(df)

    for i, row in df.iterrows():
        tipo = str(row.get("TIPO_ANTIGO", "") or "")
        objeto = str(row.get("OBJETO", "") or "")
        result = classify_row(tipo, objeto)
        results.append(result)

        if progress_callback and (i % max(1, total // 100) == 0 or i == total - 1):
            pct = min((len(results) / total), 1.0)
            progress_callback(pct, f"Processando linha {len(results):,} / {total:,}…")

    results_df = pd.DataFrame(results)
    # Reset index before concat
    df = df.reset_index(drop=True)
    results_df = results_df.reset_index(drop=True)
    return pd.concat([df, results_df], axis=1)
