from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import truststore


@dataclass(frozen=True)
class RAGConfig:
    pdf_path: Path = Path("data/CKS-Style-Conditions.pdf")
    index_dir: Path = Path("vectorstore/cks_faiss")
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 5
    min_relevance_score: float = 0.2
    llm_model: str = "llama-3.1-8b-instant"


def setup_environment() -> str:
    truststore.inject_into_ssl()
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")
    return api_key


def resolve_pdf_path(config: RAGConfig) -> Path:
    if config.pdf_path.exists():
        return config.pdf_path

    pdf_candidates = sorted(config.pdf_path.parent.glob("*.pdf"))
    if len(pdf_candidates) == 1:
        return pdf_candidates[0]
    if len(pdf_candidates) > 1:
        raise FileNotFoundError(
            f"Configured PDF not found: {config.pdf_path}. "
            "Multiple PDFs were found in data/. Keep one PDF only or name the file "
            "CKS-Style-Conditions.pdf."
        )
    raise FileNotFoundError(
        f"Configured PDF not found: {config.pdf_path}. "
        "Add your source PDF to data/ (or upload it from the app sidebar)."
    )


def load_clinical_pdf(pdf_path: Path) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(str(pdf_path)).load()
    if docs:
        return docs

    # Fallback loader for edge-case PDFs with extractor quirks.
    fallback_docs = DirectoryLoader(
        str(pdf_path.parent),
        glob=pdf_path.name,
        loader_cls=PyPDFLoader,
    ).load()
    if not fallback_docs:
        raise ValueError(f"No text extracted from {pdf_path}")
    return fallback_docs


def split_documents(documents: list[Document], config: RAGConfig) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_embeddings(config: RAGConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.embeddings_model)


def build_and_save_index(chunks: list[Document], config: RAGConfig) -> FAISS:
    embeddings = build_embeddings(config)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    config.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.index_dir))
    return vectorstore


def load_saved_index(config: RAGConfig) -> FAISS:
    embeddings = build_embeddings(config)
    return FAISS.load_local(
        str(config.index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_create_index(config: RAGConfig, rebuild: bool = False) -> tuple[FAISS, str]:
    if config.index_dir.exists() and not rebuild:
        return load_saved_index(config), "loaded"

    resolved_pdf_path = resolve_pdf_path(config)
    docs = load_clinical_pdf(resolved_pdf_path)
    chunks = split_documents(docs, config)
    return build_and_save_index(chunks, config), "built"


def _format_sources(docs: list[Document]) -> list[str]:
    sources: list[str] = []
    for doc in docs:
        page_raw = doc.metadata.get("page", None)
        page = page_raw + 1 if isinstance(page_raw, int) else "?"
        snippet = " ".join(doc.page_content.split())[:220].strip()
        sources.append(f"p.{page}: \"{snippet}\"")
    return sources


def _extract_retrieved_docs(
    vectorstore: FAISS,
    query: str,
    config: RAGConfig,
) -> list[Document]:
    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=config.retrieval_k
        )
        strong_docs = [doc for doc, score in results if score >= config.min_relevance_score]
        return strong_docs
    except Exception:
        return vectorstore.similarity_search(query, k=config.retrieval_k)


def answer_with_rag(
    vectorstore: FAISS,
    question: str,
    config: RAGConfig,
) -> dict[str, Any]:
    query = question
    retrieved_docs = _extract_retrieved_docs(vectorstore, query, config)

    if not retrieved_docs:
        return {
            "answer": (
                "I cannot find enough evidence in the provided clinical document "
                "to answer this safely."
            ),
            "sources": [],
        }

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a clinical-document QA assistant. Answer strictly and only from "
                    "the supplied context. If context is insufficient, say so clearly. "
                    "Do not invent facts. Do not provide diagnosis or personalized medical advice. "
                    "Keep the answer concise."
                ),
            ),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "Context:\n{context}\n\n"
                    "Return only the answer text."
                ),
            ),
        ]
    )

    llm = ChatGroq(model=config.llm_model, temperature=0)
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    answer_text = getattr(response, "content", str(response)).strip()

    return {"answer": answer_text, "sources": _format_sources(retrieved_docs)}


def extract_condition_candidates(
    documents: list[Document],
    max_items: int = 10,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    heading_pattern = re.compile(r"^[A-Z][A-Za-z0-9 ,/'()-]{3,80}$")

    for doc in documents:
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        for line in lines:
            if not heading_pattern.match(line):
                continue
            # Skip lines that are likely full sentences.
            if line.endswith(".") or len(line.split()) > 8:
                continue
            normalized = line.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(line)
            if len(candidates) >= max_items:
                return candidates
    return candidates
