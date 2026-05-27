# UK Clinical RAG Chatbot (Streamlit)

Minimal production-ready RAG chatbot grounded in one PDF: `data/CKS-Style-Conditions.pdf`.

## Stack

- `streamlit`
- `langchain`, `langchain-core`, `langchain-text-splitters`
- `langchain-community` (`PyPDFLoader`, `DirectoryLoader`, `Qdrant`)
- `langchain-huggingface` (`sentence-transformers/all-MiniLM-L6-v2`)
- `langchain-groq` (`ChatGroq`)
- `qdrant-client`
- `python-dotenv`
- `truststore`
- `uv`

## Setup

1. Put your clinical source at `data/CKS-Style-Conditions.pdf`.
2. Copy `.env.example` to `.env` and set:
   - `GROQ_API_KEY=...`
3. Install dependencies:
   - `uv venv`
   - `uv sync`
4. Run:
   - `uv run streamlit run app.py`

## Behavior

- Uses one PDF only for retrieval context.
- Builds Qdrant index at `vectorstore/cks_qdrant/` on first run and reuses it later.
- If context is insufficient, it explicitly says it cannot answer from the document.
- Shows page-based source snippets for each answer.

## Deploy (Streamlit Community Cloud)

1. Push this repo including `data/CKS-Style-Conditions.pdf`.
2. Set `GROQ_API_KEY` in app secrets.
3. App entrypoint: `app.py`.

Note: local Qdrant persistence is used through `qdrant-client` path storage.
