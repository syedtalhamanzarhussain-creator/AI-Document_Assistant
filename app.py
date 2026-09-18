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


APP_TITLE = "DocAI — Enterprise Document Intelligence"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Professional responsive UI ----------
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: #f7f9fc;
    }
    [data-testid="stHeader"] {
        background: rgba(247,249,252,0.92);
    }
    .block-container {
        max-width: 1180px;
        padding: 2rem 1.25rem 6rem 1.25rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e8edf3;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1.25rem 1rem 2rem 1rem;
    }

    /* Brand */
    .docai-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0 0 1.5rem 0;
    }
    .docai-logo {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #111827, #334155);
        color: white;
        font-size: 24px;
        box-shadow: 0 8px 24px rgba(15,23,42,.14);
        flex: 0 0 auto;
    }
    .docai-brand-title {
        font-size: 1.08rem;
        font-weight: 800;
        line-height: 1.1;
        color: #111827;
        margin: 0;
    }
    .docai-brand-subtitle {
        font-size: .76rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Hero */
    .docai-hero {
        padding: 2.2rem 2.2rem 2rem 2.2rem;
        border: 1px solid #e5eaf0;
        border-radius: 24px;
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        box-shadow: 0 16px 50px rgba(15,23,42,.06);
        margin-bottom: 1.25rem;
    }
    .docai-eyebrow {
        display: inline-block;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: .7rem;
    }
    .docai-hero h1 {
        font-size: clamp(2rem, 5vw, 3.35rem);
        line-height: 1.05;
        letter-spacing: -.045em;
        color: #0f172a;
        margin: 0 0 .85rem 0;
    }
    .docai-hero p {
        max-width: 760px;
        color: #475569;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 1.1rem 0 1.5rem 0;
    }
    .feature-card {
        background: #fff;
        border: 1px solid #e5eaf0;
        border-radius: 18px;
        padding: 1.1rem;
        min-height: 120px;
    }
    .feature-icon {
        font-size: 1.35rem;
        margin-bottom: .55rem;
    }
    .feature-title {
        font-weight: 750;
        color: #111827;
        margin-bottom: .25rem;
    }
    .feature-text {
        color: #64748b;
        font-size: .86rem;
        line-height: 1.45;
    }

    /* Section labels */
    .section-label {
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .07em;
        text-transform: uppercase;
        color: #64748b;
        margin: 1.2rem 0 .55rem 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 11px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #dbe2ea;
    }
    .stButton > button:hover {
        border-color: #94a3b8;
    }

    /* Chat */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: .65rem;
    }
    [data-testid="stChatInput"] {
        padding-bottom: .5rem;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e5eaf0;
        padding: .8rem;
        border-radius: 14px;
    }

    /* Footer */
    .docai-footer {
        text-align: center;
        color: #94a3b8;
        font-size: .78rem;
        padding: 2rem 0 0 0;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem .75rem 5rem .75rem;
        }
        .docai-hero {
            padding: 1.35rem 1.15rem;
            border-radius: 18px;
        }
        .docai-hero h1 {
            font-size: 2rem;
        }
        .docai-hero p {
            font-size: .94rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .feature-card {
            min-height: auto;
        }
        .docai-brand {
            margin-bottom: 1rem;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1rem .8rem 1.5rem .8rem;
        }
        [data-testid="stMetric"] {
            margin-bottom: .5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def render_brand():
    st.markdown("""
    <div class="docai-brand">
        <div class="docai-logo">📘</div>
        <div>
            <div class="docai-brand-title">DocAI</div>
            <div class="docai-brand-subtitle">Enterprise Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_hero():
    st.markdown("""
    <div class="docai-hero">
        <div class="docai-eyebrow">AI-powered document intelligence</div>
        <h1>Ask your documents.<br>Get grounded answers.</h1>
        <p>
            Upload business documents or connect a public Google Drive file or folder.
            DocAI retrieves the most relevant passages and generates answers grounded
            only in your indexed knowledge base.
        </p>
    </div>
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Multi-document</div>
            <div class="feature-text">Work with PDF, DOCX, TXT and Markdown files in one knowledge base.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔎</div>
            <div class="feature-title">Grounded retrieval</div>
            <div class="feature-text">Semantic search finds relevant document passages before every answer.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">☁️</div>
            <div class="feature-title">Google Drive</div>
            <div class="feature-text">Load public/shared Drive files or folders without manual downloading.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown(
        '<div class="docai-footer">DocAI · Enterprise Document Intelligence · '
        'Answers are grounded in your indexed documents.</div>',
        unsafe_allow_html=True
    )


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_groq_client(api_key: str):
    return Groq(api_key=api_key)


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


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    return "\n".join(paragraphs).strip()


def extract_text(file_bytes: bytes, file_name: str) -> str:
    extension = file_name.lower().split(".")[-1]

    try:
        if extension == "pdf":
            return extract_pdf_text(file_bytes)
        if extension == "docx":
            return extract_docx_text(file_bytes)
        if extension in {"txt", "md"}:
            return file_bytes.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Unsupported file type: .{extension}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not extract text from '{file_name}': {exc}"
        ) from exc


def create_chunks(text: str, source_file: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text_chunks = splitter.split_text(text)
    chunks = []

    for chunk_id, chunk_text in enumerate(text_chunks):
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


def build_faiss_index(chunks: List[Dict], embedding_model) -> faiss.Index:
    if not chunks:
        raise ValueError("No chunks are available for indexing.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]

    # Inner product on normalized embeddings is cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    embedding_model,
    top_k: int = TOP_K,
) -> List[Tuple[Dict, float]]:
    if index is None or index.ntotal == 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position >= 0:
            results.append((chunks[index_position], float(score)))

    return results


def build_grounded_prompt(
    query: str,
    retrieved_chunks: List[Tuple[Dict, float]],
) -> str:
    context_parts = []

    for position, (chunk, _score) in enumerate(
        retrieved_chunks, start=1
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

    context = "\n".join(context_parts)

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


def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[Dict, float]],
    groq_client: Groq,
) -> str:
    if not retrieved_chunks:
        return "Information not found in the uploaded documents."

    prompt = build_grounded_prompt(query, retrieved_chunks)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strictly grounded document question-answering "
                    "system. Never use knowledge outside the supplied context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1200,
    )

    answer = response.choices[0].message.content
    return (
        answer.strip()
        if answer and answer.strip()
        else "Information not found in the uploaded documents."
    )


def process_file_items(file_items) -> None:
    """Process local files represented as (file_name, file_bytes)."""
    all_chunks = []
    documents = []
    progress = st.progress(0)
    total_files = len(file_items)

    for file_number, (file_name, file_bytes) in enumerate(
        file_items, start=1
    ):
        try:
            if not file_bytes:
                st.warning(f"'{file_name}' is empty and was skipped.")
                continue

            text = extract_text(file_bytes, file_name)

            if not text.strip():
                st.warning(
                    f"No readable text was found in '{file_name}'."
                )
                continue

            chunks = create_chunks(text, file_name)

            if not chunks:
                st.warning(
                    f"No chunks could be created from '{file_name}'."
                )
                continue

            documents.append({"file_name": file_name, "text": text})
            all_chunks.extend(chunks)

        except Exception as exc:
            st.error(f"Error processing '{file_name}': {exc}")

        progress.progress(file_number / total_files)

    progress.empty()

    if not all_chunks:
        st.error("No usable text was found in the documents.")
        return

    try:
        embedding_model = load_embedding_model()

        with st.spinner(
            "Generating embeddings and building FAISS index..."
        ):
            index = build_faiss_index(all_chunks, embedding_model)

        st.session_state.documents = documents
        st.session_state.chunks = all_chunks
        st.session_state.index = index
        st.session_state.indexed_file_names = [
            document["file_name"] for document in documents
        ]
        st.session_state.chat_history = []

        st.success(
            f"Indexed {len(documents)} document(s) "
            f"into {len(all_chunks)} chunks."
        )

    except Exception as exc:
        st.error(f"Could not build the vector index: {exc}")


def process_uploaded_files(uploaded_files) -> None:
    file_items = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    ]
    process_file_items(file_items)


def download_google_drive_files(drive_url: str):
    """
    Download a public/shared Google Drive file or folder into a temporary
    directory and return a list of (display_name, bytes).

    The Drive item must be shared as:
    Anyone with the link -> Viewer.
    """
    drive_url = drive_url.strip()

    if not drive_url:
        raise ValueError("Please paste a Google Drive file or folder link.")

    if "drive.google.com" not in drive_url and "docs.google.com" not in drive_url:
        raise ValueError(
            "Please provide a valid Google Drive or Google Docs share link."
        )

    temp_dir = tempfile.mkdtemp(prefix="rag_drive_")

    try:
        downloaded_paths = []

        with st.spinner("Downloading documents from Google Drive..."):
            # Folder links are automatically recognized by gdown.
            if "/folders/" in drive_url:
                folder_paths = gdown.download_folder(
                    url=drive_url,
                    output=temp_dir,
                    quiet=True,
                    use_cookies=False,
                )

                if folder_paths:
                    downloaded_paths.extend(folder_paths)

            else:
                # output=temp_dir lets gdown determine the original filename.
                downloaded_path = gdown.download(
                    url=drive_url,
                    output=temp_dir,
                    quiet=True,
                )

                if downloaded_path:
                    downloaded_paths.append(downloaded_path)

        if not downloaded_paths:
            raise RuntimeError(
                "Google Drive did not return any downloadable files."
            )

        file_items = []
        supported_count = 0

        for raw_path in downloaded_paths:
            path = Path(raw_path)

            if not path.is_file():
                continue

            extension = path.suffix.lower().lstrip(".")
            if extension not in SUPPORTED_TYPES:
                continue

            try:
                file_items.append((path.name, path.read_bytes()))
                supported_count += 1
            except Exception as exc:
                st.warning(
                    f"Could not read '{path.name}' from the downloaded Drive data: {exc}"
                )

        if not file_items:
            raise RuntimeError(
                "No supported documents were found. "
                "The Drive file/folder must contain PDF, DOCX, TXT, or MD files."
            )

        return file_items

    except Exception:
        raise
    finally:
        # The files are already in memory, so remove the temporary downloads.
        shutil.rmtree(temp_dir, ignore_errors=True)


with st.sidebar:
    render_brand()
    st.markdown('<div class="section-label">Knowledge base</div>', unsafe_allow_html=True)
    st.caption("Build a private, session-based knowledge base from your documents.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT and Markdown.",
    )

    if uploaded_files and st.button(
        "🔄 Process Uploaded Documents",
        use_container_width=True,
    ):
        process_uploaded_files(uploaded_files)

    st.divider()

    st.subheader("☁️ Google Drive")
    st.caption(
        "Use a public/shared Google Drive file or folder. "
        "Set sharing to 'Anyone with the link → Viewer'."
    )

    drive_url = st.text_input(
        "Google Drive link",
        placeholder="https://drive.google.com/drive/folders/...",
        help=(
            "Paste a Google Drive file/folder link. "
            "The link must be accessible to anyone with the link."
        ),
    )

    if st.button(
        "☁️ Load from Google Drive",
        use_container_width=True,
    ):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive link first.")
        else:
            try:
                drive_file_items = download_google_drive_files(drive_url)

                st.success(
                    f"Downloaded {len(drive_file_items)} supported "
                    f"document(s) from Google Drive."
                )

                process_file_items(drive_file_items)

            except Exception as exc:
                st.error(
                    "Could not load the Google Drive documents. "
                    f"Details: {exc}"
                )

    st.divider()
    st.subheader("Knowledge Base")

    if st.session_state.index is not None:
        st.metric("Documents", len(st.session_state.documents))
        st.metric("Text Chunks", len(st.session_state.chunks))
        st.caption("Indexed files:")
        for file_name in st.session_state.indexed_file_names:
            st.write(f"• {file_name}")
    else:
        st.info("No documents indexed yet.")

    st.divider()

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


render_hero()

groq_api_key = None
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.warning(
        "Groq API key is missing. Add GROQ_API_KEY to "
        ".streamlit/secrets.toml."
    )
    st.stop()

try:
    groq_client = load_groq_client(groq_api_key)
except Exception as exc:
    st.error(f"Could not initialize Groq client: {exc}")
    st.stop()


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📖 Sources Used"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
**Source:** `{source["source_file"]}`  
**Chunk ID:** `{source["chunk_id"]}`  
**Similarity:** `{source["score"]:.4f}`

> {source["text"]}
"""
                    )


if st.session_state.index is None:
    st.info(
        "Upload/process documents or load documents from Google Drive "
        "before asking a question."
    )
else:
    query = st.chat_input("Ask a question about your documents...")

    if query:
        query = query.strip()

        if not query:
            st.warning("Please enter a question.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.chat_history.append(
            {"role": "user", "content": query}
        )

        try:
            embedding_model = load_embedding_model()

            with st.spinner("Searching the knowledge base..."):
                retrieved_chunks = retrieve_chunks(
                    query=query,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    embedding_model=embedding_model,
                    top_k=TOP_K,
                )
        except Exception as exc:
            st.error(f"Error during vector search: {exc}")
            st.stop()

        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating grounded answer..."):
                    answer = generate_answer(
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        groq_client=groq_client,
                    )

                st.markdown(answer)

                if retrieved_chunks:
                    with st.expander("📖 Sources Used"):
                        for chunk, score in retrieved_chunks:
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
                    "An error occurred while generating "
                    f"the answer: {exc}"
                )
                st.error(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {**chunk, "score": score}
                    for chunk, score in retrieved_chunks
                ],
            }
        )

render_footer()
