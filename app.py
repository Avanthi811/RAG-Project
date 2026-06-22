import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from cryptography.fernet import Fernet
import os

# NOTE: In your deployed app, you should reconstruct the same pipeline.
# For simplicity here, we assume objects 'retriever', 'llm', and 'rag_chain', 'ask_bot'
# are available or re-created similarly.

st.title("Zyro Dynamics HR Help Desk")

st.write("Ask any question about Zyro Dynamics HR policies.")

user_question = st.text_input("Your question")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.button("Ask") and user_question.strip():
    try:
        result = ask_bot(user_question)
        answer = result["answer"]
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", answer))
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("### Conversation")
for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")

with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py created.")
