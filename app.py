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


APP_TITLE = "Enterprise Multi-Document RAG Assistant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


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
    st.header("📚 Document Knowledge Base")
    st.caption(
        "Upload documents or load public/shared Google Drive documents "
        "to create a temporary session-based RAG knowledge base."
    )

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


st.title("📚 Enterprise Multi-Document RAG Assistant")
st.write(
    "Ask questions about your uploaded documents. "
    "Answers are grounded exclusively in retrieved document context."
)

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
