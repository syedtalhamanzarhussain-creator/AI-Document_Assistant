import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq


# =========================================================
# CONFIGURATION
# =========================================================

APP_TITLE = "DocAI — Enterprise Document Intelligence"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GROQ_MODEL_NAME = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 5

SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "txt",
    "md",
]


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown(
    """
<style>

/* =======================================================
   GLOBAL THEME
   ======================================================= */

:root {
    color-scheme: light !important;

    --docai-bg: #f7f9fc;
    --docai-surface: #ffffff;
    --docai-surface-soft: #f8fafc;

    --docai-border: #e2e8f0;

    --docai-text: #0f172a;
    --docai-muted: #64748b;

    --docai-primary: #2563eb;
    --docai-primary-dark: #1d4ed8;
}


/* Streamlit theme variables */

html {
    --st-background-color: #f7f9fc !important;
    --st-secondary-background-color: #ffffff !important;
    --st-text-color: #0f172a !important;
    --st-border-color: #e2e8f0 !important;
    --st-primary-color: #2563eb !important;
}


html,
body {
    color-scheme: light !important;
}


body {
    background: #f7f9fc !important;
}


/* =======================================================
   APP BACKGROUND
   ======================================================= */

[data-testid="stAppViewContainer"] {
    background: #f7f9fc !important;
    color: #0f172a !important;
}


[data-testid="stMain"] {
    background: #f7f9fc !important;
}


.block-container {
    max-width: 1180px !important;

    padding-top: 1.5rem !important;
    padding-bottom: 6rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}


/* =======================================================
   SIDEBAR
   ======================================================= */

section[data-testid="stSidebar"] {
    background: #ffffff !important;

    border-right: 1px solid #e2e8f0 !important;
}


section[data-testid="stSidebar"] > div {
    background: #ffffff !important;
}


section[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem 2rem 1rem !important;
}


/* =======================================================
   GENERAL TEXT
   ======================================================= */

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {
    color: #0f172a;
}


[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #0f172a;
}


[data-testid="stCaptionContainer"] * {
    color: #64748b !important;
}


/* =======================================================
   SIDEBAR BRAND
   ======================================================= */

.docai-brand {
    display: flex;

    align-items: center;

    gap: 12px;

    margin-bottom: 1.5rem;
}


.docai-logo {
    width: 46px;

    height: 46px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

    background: #0f172a;

    color: #ffffff !important;

    font-size: 23px;

    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
}


.docai-brand-title {
    color: #0f172a !important;

    font-size: 1.08rem;

    font-weight: 800;

    line-height: 1.1;
}


.docai-brand-subtitle {
    color: #64748b !important;

    font-size: 0.76rem;

    margin-top: 4px;
}


.section-label {
    color: #64748b !important;

    font-size: 0.72rem;

    font-weight: 800;

    letter-spacing: 0.09em;

    text-transform: uppercase;

    margin-top: 1rem;

    margin-bottom: 0.55rem;
}


/* =======================================================
   HERO
   ======================================================= */

.docai-hero {
    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            #eef4ff 100%
        );

    border: 1px solid #e2e8f0;

    border-radius: 24px;

    padding: 2.35rem 2.25rem 2.1rem;

    margin-bottom: 1rem;

    box-shadow:
        0 16px 50px rgba(15, 23, 42, 0.06);
}


.docai-eyebrow {
    color: #2563eb !important;

    font-size: 0.73rem;

    font-weight: 800;

    letter-spacing: 0.1em;

    text-transform: uppercase;

    margin-bottom: 0.7rem;
}


.docai-hero h1 {
    color: #0f172a !important;

    font-size: clamp(2rem, 5vw, 3.2rem);

    line-height: 1.05;

    letter-spacing: -0.045em;

    margin: 0 0 0.9rem 0;

    font-weight: 800;
}


.docai-hero p {
    color: #475569 !important;

    max-width: 780px;

    font-size: 1rem;

    line-height: 1.65;

    margin: 0;
}


/* =======================================================
   FEATURE CARDS
   ======================================================= */

.feature-grid {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 14px;

    margin-bottom: 1.5rem;
}


.feature-card {
    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 18px;

    padding: 1.15rem;

    min-height: 125px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.035);
}


.feature-icon {
    font-size: 1.35rem;

    margin-bottom: 0.55rem;
}


.feature-title {
    color: #0f172a !important;

    font-weight: 800;

    font-size: 0.96rem;

    margin-bottom: 0.3rem;
}


.feature-text {
    color: #64748b !important;

    font-size: 0.84rem;

    line-height: 1.5;
}


/* =======================================================
   INPUTS
   ======================================================= */

[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] {
    background: #ffffff !important;

    border-color: #cbd5e1 !important;
}


[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] input {
    background: #ffffff !important;

    color: #0f172a !important;

    -webkit-text-fill-color: #0f172a !important;
}


[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder {
    color: #94a3b8 !important;

    -webkit-text-fill-color: #94a3b8 !important;

    opacity: 1 !important;
}


/* =======================================================
   FILE UPLOADER
   ======================================================= */

[data-testid="stFileUploader"] section {
    background: #ffffff !important;

    border: 1px dashed #cbd5e1 !important;

    border-radius: 14px !important;
}


[data-testid="stFileUploader"] section * {
    color: #0f172a !important;
}


/* =======================================================
   BUTTONS
   ======================================================= */

.stButton > button {
    min-height: 44px !important;

    border-radius: 11px !important;

    border: 1px solid #dbe2ea !important;

    background: #ffffff !important;

    color: #0f172a !important;

    font-weight: 700 !important;

    box-shadow: none !important;
}


.stButton > button:hover {
    background: #f8fafc !important;

    border-color: #94a3b8 !important;

    color: #0f172a !important;
}


/* =======================================================
   METRICS
   ======================================================= */

[data-testid="stMetric"] {
    background: #ffffff !important;

    border: 1px solid #e2e8f0 !important;

    border-radius: 14px !important;

    padding: 0.8rem !important;
}


[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"] * {
    color: #0f172a !important;
}


/* =======================================================
   ALERTS
   ======================================================= */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}


[data-testid="stAlert"] * {
    color: #0f172a !important;
}


/* =======================================================
   CHAT
   ======================================================= */

[data-testid="stChatMessage"] {
    border-radius: 16px !important;

    margin-bottom: 0.65rem !important;
}


[data-testid="stChatMessage"] * {
    color: #0f172a !important;
}


[data-testid="stChatInput"] {
    background: #ffffff !important;

    border-top: 1px solid #e2e8f0 !important;
}


[data-testid="stChatInput"] > div {
    background: #ffffff !important;
}


[data-testid="stChatInput"] textarea {
    background: #ffffff !important;

    color: #0f172a !important;

    -webkit-text-fill-color: #0f172a !important;

    border-color: #cbd5e1 !important;
}


[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;

    -webkit-text-fill-color: #64748b !important;
}


/* =======================================================
   EXPANDERS
   ======================================================= */

[data-testid="stExpander"] {
    background: #ffffff !important;

    border: 1px solid #e2e8f0 !important;

    border-radius: 14px !important;
}


[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #0f172a !important;
}


/* =======================================================
   DIVIDER
   ======================================================= */

hr {
    border-color: #e2e8f0 !important;
}


/* =======================================================
   FOOTER
   ======================================================= */

.docai-footer {
    color: #94a3b8 !important;

    text-align: center;

    font-size: 0.78rem;

    padding-top: 2rem;

    padding-bottom: 1rem;
}


/* =======================================================
   MOBILE
   ======================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-top: 1rem !important;

        padding-left: 0.75rem !important;

        padding-right: 0.75rem !important;
    }


    .docai-hero {
        padding: 1.4rem 1.15rem;

        border-radius: 18px;
    }


    .docai-hero h1 {
        font-size: 2rem;
    }


    .docai-hero p {
        font-size: 0.9rem;
    }


    .feature-grid {
        grid-template-columns: 1fr;

        gap: 10px;
    }


    .feature-card {
        min-height: auto;
    }


    section[data-testid="stSidebar"] .block-container {
        padding-left: 0.8rem !important;

        padding-right: 0.8rem !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HTML UI HELPERS
# =========================================================

def render_brand():
    st.html(
        """
<div class="docai-brand">
    <div class="docai-logo">📘</div>

    <div>
        <div class="docai-brand-title">
            DocAI
        </div>

        <div class="docai-brand-subtitle">
            Enterprise Document Intelligence
        </div>
    </div>
</div>
"""
    )


def render_hero():
    st.html(
        """
<div class="docai-hero">

    <div class="docai-eyebrow">
        AI-powered document intelligence
    </div>

    <h1>
        Ask your documents.<br>
        Get grounded answers.
    </h1>

    <p>
        Upload business documents or connect a public
        Google Drive file or folder. DocAI retrieves
        the most relevant passages and generates answers
        grounded only in your indexed knowledge base.
    </p>

</div>


<div class="feature-grid">

    <div class="feature-card">

        <div class="feature-icon">
            📄
        </div>

        <div class="feature-title">
            Multi-document
        </div>

        <div class="feature-text">
            Work with PDF, DOCX, TXT and Markdown
            files in one knowledge base.
        </div>

    </div>


    <div class="feature-card">

        <div class="feature-icon">
            🔎
        </div>

        <div class="feature-title">
            Grounded retrieval
        </div>

        <div class="feature-text">
            Semantic search finds relevant document
            passages before every answer.
        </div>

    </div>


    <div class="feature-card">

        <div class="feature-icon">
            ☁️
        </div>

        <div class="feature-title">
            Google Drive
        </div>

        <div class="feature-text">
            Load public/shared Drive files or folders
            without manual downloading.
        </div>

    </div>

</div>
"""
    )


def render_footer():
    st.html(
        """
<div class="docai-footer">
    DocAI · Enterprise Document Intelligence ·
    Answers are grounded in your indexed documents.
</div>
"""
    )


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session_state():

    defaults = {
        "documents": [],
        "chunks": [],
        "index": None,
        "chat_history": [],
        "indexed_file_names": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# =========================================================
# CACHED MODELS
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@st.cache_resource
def load_groq_client(api_key: str):

    return Groq(
        api_key=api_key
    )


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(
    file_bytes: bytes
) -> str:

    reader = PdfReader(
        BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:

            pages.append(
                text
            )

    return "\n\n".join(
        pages
    ).strip()


# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx_text(
    file_bytes: bytes
) -> str:

    document = Document(
        BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(
                text
            )

    return "\n".join(
        paragraphs
    ).strip()


# =========================================================
# GENERIC TEXT EXTRACTION
# =========================================================

def extract_text(
    file_bytes: bytes,
    file_name: str,
) -> str:

    extension = (
        file_name
        .lower()
        .split(".")[-1]
    )

    try:

        if extension == "pdf":

            return extract_pdf_text(
                file_bytes
            )

        elif extension == "docx":

            return extract_docx_text(
                file_bytes
            )

        elif extension in {
            "txt",
            "md",
        }:

            return file_bytes.decode(
                "utf-8",
                errors="replace",
            ).strip()

        else:

            raise ValueError(
                f"Unsupported file type: .{extension}"
            )

    except Exception as exc:

        raise RuntimeError(
            f"Could not extract text from "
            f"'{file_name}': {exc}"
        ) from exc


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text: str,
    source_file: str,
) -> List[Dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    text_chunks = splitter.split_text(
        text
    )

    chunks = []

    for chunk_id, chunk_text in enumerate(
        text_chunks
    ):

        cleaned_text = chunk_text.strip()

        if cleaned_text:

            chunks.append(
                {
                    "text": cleaned_text,

                    "source_file": source_file,

                    "chunk_id": chunk_id,
                }
            )

    return chunks


# =========================================================
# BUILD FAISS INDEX
# =========================================================

def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
):

    if not chunks:

        raise ValueError(
            "No chunks are available for indexing."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# RETRIEVE CHUNKS
# =========================================================

def retrieve_chunks(
    query: str,
    index,
    chunks: List[Dict],
    embedding_model,
    top_k: int = TOP_K,
):

    if (
        index is None
        or index.ntotal == 0
    ):

        return []

    query_embedding = embedding_model.encode(
        [query],

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    k = min(
        top_k,
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position >= 0:

            results.append(
                (
                    chunks[index_position],
                    float(score),
                )
            )

    return results


# =========================================================
# GROUNDED PROMPT
# =========================================================

def build_grounded_prompt(
    query: str,
    retrieved_chunks,
) -> str:

    context_parts = []

    for position, (
        chunk,
        _score,
    ) in enumerate(
        retrieved_chunks,
        start=1,
    ):

        context_parts.append(
            f"""
--- CONTEXT SNIPPET {position} ---
Source File: {chunk["source_file"]}
Chunk ID: {chunk["chunk_id"]}

{chunk["text"]}
--- END CONTEXT SNIPPET ---
"""
        )

    context = "\n".join(
        context_parts
    )

    return f"""
You are an enterprise document question-answering assistant.

Answer the user's question using ONLY the context snippets supplied below.

STRICT GROUNDING RULES:

1. Use ONLY information contained in the supplied context snippets.

2. Do NOT use your pretrained knowledge.

3. Do NOT use outside information.

4. Do NOT make assumptions.

5. Do NOT guess.

6. If the answer cannot be found in the context, respond EXACTLY:

Information not found in the uploaded documents.

7. You may combine information from multiple snippets when supported.

8. Keep the answer clear and concise.

9. Do not mention information that is not supported by the context.

CONTEXT:

{context}

USER QUESTION:

{query}
"""


# =========================================================
# GROQ GENERATION
# =========================================================

def generate_answer(
    query: str,
    retrieved_chunks,
    groq_client,
) -> str:

    if not retrieved_chunks:

        return (
            "Information not found in the uploaded documents."
        )

    prompt = build_grounded_prompt(
        query,

        retrieved_chunks,
    )

    response = groq_client.chat.completions.create(

        model=GROQ_MODEL_NAME,

        messages=[
            {
                "role": "system",

                "content": (
                    "You are a strictly grounded "
                    "document question-answering system. "
                    "Never use knowledge outside the "
                    "supplied context."
                ),
            },

            {
                "role": "user",

                "content": prompt,
            },
        ],

        temperature=0,

        max_tokens=1200,
    )

    answer = response.choices[0].message.content

    if not answer:

        return (
            "Information not found in the uploaded documents."
        )

    answer = answer.strip()

    if not answer:

        return (
            "Information not found in the uploaded documents."
        )

    return answer


# =========================================================
# PROCESS DOCUMENT FILES
# =========================================================

def process_file_items(
    file_items,
) -> None:

    all_chunks = []

    documents = []

    total_files = len(
        file_items
    )

    progress = st.progress(
        0
    )

    for file_number, (
        file_name,
        file_bytes,
    ) in enumerate(
        file_items,
        start=1,
    ):

        try:

            if not file_bytes:

                st.warning(
                    f"'{file_name}' is empty and was skipped."
                )

                continue

            text = extract_text(
                file_bytes,

                file_name,
            )

            if not text.strip():

                st.warning(
                    f"No readable text was found "
                    f"in '{file_name}'."
                )

                continue

            chunks = create_chunks(
                text,

                file_name,
            )

            if not chunks:

                st.warning(
                    f"No chunks could be created "
                    f"from '{file_name}'."
                )

                continue

            documents.append(
                {
                    "file_name": file_name,

                    "text": text,
                }
            )

            all_chunks.extend(
                chunks
            )

        except Exception as exc:

            st.error(
                f"Error processing "
                f"'{file_name}': {exc}"
            )

        finally:

            if total_files:

                progress.progress(
                    file_number / total_files
                )

    progress.empty()

    if not all_chunks:

        st.error(
            "No usable text was found in the documents."
        )

        return

    try:

        embedding_model = (
            load_embedding_model()
        )

        with st.spinner(
            "Generating embeddings and "
            "building FAISS index..."
        ):

            index = build_faiss_index(
                all_chunks,

                embedding_model,
            )

        st.session_state.documents = (
            documents
        )

        st.session_state.chunks = (
            all_chunks
        )

        st.session_state.index = (
            index
        )

        st.session_state.indexed_file_names = [
            document["file_name"]

            for document in documents
        ]

        st.session_state.chat_history = []

        st.success(
            f"Indexed {len(documents)} "
            f"document(s) into "
            f"{len(all_chunks)} chunks."
        )

    except Exception as exc:

        st.error(
            f"Could not build the vector index: {exc}"
        )


# =========================================================
# LOCAL UPLOAD PROCESSING
# =========================================================

def process_uploaded_files(
    uploaded_files,
) -> None:

    file_items = []

    for uploaded_file in uploaded_files:

        file_items.append(
            (
                uploaded_file.name,

                uploaded_file.getvalue(),
            )
        )

    process_file_items(
        file_items
    )


# =========================================================
# GOOGLE DRIVE
# =========================================================

def download_google_drive_files(
    drive_url: str,
):

    drive_url = drive_url.strip()

    if not drive_url:

        raise ValueError(
            "Please paste a Google Drive "
            "file or folder link."
        )

    if (
        "drive.google.com" not in drive_url
        and "docs.google.com" not in drive_url
    ):

        raise ValueError(
            "Please provide a valid Google Drive "
            "or Google Docs share link."
        )

    temp_dir = tempfile.mkdtemp(
        prefix="rag_drive_"
    )

    try:

        downloaded_paths = []

        with st.spinner(
            "Downloading documents from Google Drive..."
        ):

            if "/folders/" in drive_url:

                folder_paths = (
                    gdown.download_folder(
                        url=drive_url,

                        output=temp_dir,

                        quiet=True,

                        use_cookies=False,
                    )
                )

                if folder_paths:

                    downloaded_paths.extend(
                        folder_paths
                    )

            else:

                downloaded_path = (
                    gdown.download(
                        url=drive_url,

                        output=temp_dir,

                        quiet=True,
                    )
                )

                if downloaded_path:

                    downloaded_paths.append(
                        downloaded_path
                    )

        if not downloaded_paths:

            raise RuntimeError(
                "Google Drive did not return "
                "any downloadable files."
            )

        file_items = []

        for raw_path in downloaded_paths:

            path = Path(
                raw_path
            )

            if not path.is_file():

                continue

            extension = (
                path.suffix
                .lower()
                .lstrip(".")
            )

            if extension not in SUPPORTED_TYPES:

                continue

            try:

                file_items.append(
                    (
                        path.name,

                        path.read_bytes(),
                    )
                )

            except Exception as exc:

                st.warning(
                    f"Could not read "
                    f"'{path.name}': {exc}"
                )

        if not file_items:

            raise RuntimeError(
                "No supported documents were found. "
                "The Drive file/folder must contain "
                "PDF, DOCX, TXT, or MD files."
            )

        return file_items

    finally:

        shutil.rmtree(
            temp_dir,

            ignore_errors=True,
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_brand()


    # -----------------------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">Knowledge base</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Build a session-based knowledge base "
        "from your documents."
    )


    uploaded_files = st.file_uploader(
        "Upload documents",

        type=SUPPORTED_TYPES,

        accept_multiple_files=True,

        help=(
            "Supported formats: PDF, DOCX, "
            "TXT and Markdown."
        ),
    )


    if uploaded_files:

        if st.button(
            "🔄 Process Uploaded Documents",

            use_container_width=True,
        ):

            process_uploaded_files(
                uploaded_files
            )


    st.divider()


    # -----------------------------------------------------
    # GOOGLE DRIVE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">Google Drive</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Use a public/shared Google Drive file "
        "or folder. Sharing must be "
        "'Anyone with the link → Viewer'."
    )


    drive_url = st.text_input(
        "Google Drive link",

        placeholder=(
            "https://drive.google.com/drive/folders/..."
        ),

        help=(
            "Paste a public/shared Google Drive "
            "or Google Docs link."
        ),
    )


    if st.button(
        "☁️ Load from Google Drive",

        use_container_width=True,
    ):

        if not drive_url.strip():

            st.warning(
                "Please paste a Google Drive link first."
            )

        else:

            try:

                drive_file_items = (
                    download_google_drive_files(
                        drive_url
                    )
                )

                st.success(
                    f"Downloaded "
                    f"{len(drive_file_items)} "
                    f"supported document(s) "
                    f"from Google Drive."
                )

                process_file_items(
                    drive_file_items
                )

            except Exception as exc:

                st.error(
                    "Could not load the Google Drive "
                    f"documents.\n\n{exc}"
                )


    st.divider()


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-label">Knowledge base status</div>',
        unsafe_allow_html=True,
    )


    if st.session_state.index is not None:

        st.metric(
            "Documents",

            len(
                st.session_state.documents
            ),
        )

        st.metric(
            "Text Chunks",

            len(
                st.session_state.chunks
            ),
        )

        st.caption(
            "Indexed files:"
        )

        for file_name in (
            st.session_state.indexed_file_names
        ):

            st.write(
                f"• {file_name}"
            )

    else:

        st.info(
            "No documents indexed yet."
        )


    st.divider()


    # -----------------------------------------------------
    # CLEAR
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Knowledge Base",

        use_container_width=True,
    ):

        st.session_state.documents = []

        st.session_state.chunks = []

        st.session_state.index = None

        st.session_state.indexed_file_names = []

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN HERO
# =========================================================

render_hero()


# =========================================================
# GROQ API KEY
# =========================================================

groq_api_key = None

try:

    groq_api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

except Exception:

    groq_api_key = None


if not groq_api_key:

    st.warning(
        "Groq API key is missing. "
        "Add GROQ_API_KEY to your "
        "Streamlit secrets."
    )

    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

try:

    groq_client = load_groq_client(
        groq_api_key
    )

except Exception as exc:

    st.error(
        f"Could not initialize Groq client: {exc}"
    )

    st.stop()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in (
    st.session_state.chat_history
):

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )

        sources = message.get(
            "sources",
            [],
        )

        if (
            role == "assistant"
            and sources
        ):

            with st.expander(
                "📖 Sources Used"
            ):

                for source in sources:

                    st.markdown(
                        f"""
**Source:** `{source["source_file"]}`

**Chunk ID:** `{source["chunk_id"]}`

**Similarity:** `{source["score"]:.4f}`

> {source["text"]}
"""
                    )


# =========================================================
# INITIAL STATE MESSAGE
# =========================================================

if st.session_state.index is None:

    st.info(
        "Upload/process documents or load documents "
        "from Google Drive before asking a question."
    )


# =========================================================
# CHAT INPUT
# =========================================================

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    query = query.strip()

    if not query:

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )


    st.session_state.chat_history.append(
        {
            "role": "user",

            "content": query,
        }
    )


    # -----------------------------------------------------
    # MAKE SURE KNOWLEDGE BASE EXISTS
    # -----------------------------------------------------

    if st.session_state.index is None:

        answer = (
            "Please upload or load documents "
            "before asking a question."
        )

        with st.chat_message(
            "assistant"
        ):

            st.warning(
                answer
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",

                "content": answer,

                "sources": [],
            }
        )

        st.stop()


    # -----------------------------------------------------
    # EMBEDDING MODEL
    # -----------------------------------------------------

    try:

        embedding_model = (
            load_embedding_model()
        )

    except Exception as exc:

        st.error(
            f"Could not load embedding model: {exc}"
        )

        st.stop()


    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Searching the knowledge base..."
        ):

            retrieved_chunks = (
                retrieve_chunks(
                    query=query,

                    index=(
                        st.session_state.index
                    ),

                    chunks=(
                        st.session_state.chunks
                    ),

                    embedding_model=(
                        embedding_model
                    ),

                    top_k=TOP_K,
                )
            )

    except Exception as exc:

        st.error(
            f"Error during vector search: {exc}"
        )

        st.stop()


    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "Generating grounded answer..."
            ):

                answer = generate_answer(
                    query=query,

                    retrieved_chunks=(
                        retrieved_chunks
                    ),

                    groq_client=(
                        groq_client
                    ),
                )


            st.markdown(
                answer
            )


            # ---------------------------------------------
            # SOURCES
            # ---------------------------------------------

            if retrieved_chunks:

                with st.expander(
                    "📖 Sources Used"
                ):

                    for (
                        chunk,
                        score,
                    ) in retrieved_chunks:

                        st.markdown(
                            f"""
**Source:** `{chunk["source_file"]}`

**Chunk ID:** `{chunk["chunk_id"]}`

**Similarity:** `{score:.4f}`

> {chunk["text"]}
"""
                        )


        except Exception as exc:

            answer = (
                "An error occurred while "
                f"generating the answer: {exc}"
            )

            st.error(
                answer
            )


    # -----------------------------------------------------
    # SAVE ANSWER
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",

            "content": answer,

            "sources": [
                {
                    "source_file": chunk[
                        "source_file"
                    ],

                    "chunk_id": chunk[
                        "chunk_id"
                    ],

                    "text": chunk[
                        "text"
                    ],

                    "score": score,
                }

                for chunk, score
                in retrieved_chunks
            ],
        }
    )


# =========================================================
# FOOTER
# =========================================================

render_footer()
