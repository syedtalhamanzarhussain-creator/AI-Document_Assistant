# AI-Document_Assistant

Enterprise Multi-Document RAG Assistant using Streamlit, FAISS,
Sentence Transformers, and Groq.

## Supported document sources

- Local upload: PDF, DOCX, TXT, Markdown
- Public/shared Google Drive file links
- Public/shared Google Drive folder links

## Google Drive requirement

The Drive file or folder must be shared as:

**Anyone with the link → Viewer**

No Google OAuth is required for this public/shared-link version.

## Deploy on Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to GitHub.
2. Deploy `app.py`.
3. Select Python 3.12.
4. Add your real Groq key in Streamlit Secrets:

```toml
GROQ_API_KEY = "gsk_YOUR_REAL_GROQ_KEY"
```

5. Do not commit your real API key to GitHub.

## Google Drive workflow

Paste a Google Drive file or folder link in the sidebar and click
**Load from Google Drive**.

Supported downloaded document types:
PDF, DOCX, TXT, and MD.

Google Drive folders are downloaded recursively by gdown. The current
gdown implementation supports public/shared Drive files and folders.


## Professional UI
This version adds a responsive DocAI brand, product-style landing screen, feature cards, polished controls, and mobile-friendly CSS while preserving the RAG pipeline and Google Drive support.
