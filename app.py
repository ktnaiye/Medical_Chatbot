from __future__ import annotations

import streamlit as st

from rag import RAGConfig, answer_with_rag, get_or_create_index, setup_environment

st.set_page_config(page_title="UK Clinical RAG Chatbot", page_icon=":stethoscope:")
st.title("UK Clinical RAG Chatbot")
st.caption("Grounded QA over a single clinical PDF using LangChain + FAISS + Groq.")

config = RAGConfig()

if "messages" not in st.session_state:
    st.session_state.messages = []


def initialize_rag(rebuild: bool = False) -> tuple[bool, str]:
    try:
        setup_environment()
        index, status = get_or_create_index(config, rebuild=rebuild)
        st.session_state.vectorstore = index
        return True, status
    except Exception as exc:
        st.session_state.init_error = str(exc)
        return False, "error"


if "vectorstore" not in st.session_state:
    ok, _status = initialize_rag(rebuild=False)
    if not ok:
        st.error(st.session_state.get("init_error", "Failed to initialize."))
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about the clinical document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_with_rag(
                st.session_state.vectorstore,
                question=question,
                config=config,
            )
        st.markdown(result["answer"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
