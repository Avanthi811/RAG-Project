app_code = """
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CORPUS_PATH = "./zyro-dynamics-hr-corpus"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢"
)

st.title("🏢 Zyro Dynamics HR Help Desk")

# --------------------------------------------------
# API KEY
# --------------------------------------------------

groq_key = (
    st.secrets.get("GROQ_API_KEY")
    or os.environ.get("GROQ_API_KEY")
)

if not groq_key:
    st.error("GROQ_API_KEY not found")
    st.stop()

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=512,
    api_key=groq_key
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_template(
'''
You are the official HR Assistant for Zyro Dynamics.

Rules:

1. Answer ONLY from the provided context.
2. Do not use external knowledge.
3. Extract exact values, dates, durations and percentages.
4. Mention policy source whenever possible.
5. If answer is unavailable, respond exactly:

I cannot answer this based on the available HR policy documents.

Context:
{context}

Question:
{question}

Answer:
'''
)

# --------------------------------------------------
# OUT OF SCOPE CHECK
# --------------------------------------------------

OOS_PROMPT = ChatPromptTemplate.from_template(
'''
Determine whether the question is related to Zyro Dynamics HR policies.

Respond ONLY with:

IN_SCOPE

or

OUT_OF_SCOPE

Question:
{question}
'''
)

# --------------------------------------------------
# VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def build_rag():

    docs = []

    for file in os.listdir(CORPUS_PATH):

        if file.endswith(".pdf"):

            loader = PyPDFLoader(
                os.path.join(CORPUS_PATH, file)
            )

            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["file_name"] = file

            docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    return retriever

retriever = build_rag()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def format_docs(docs):

    return "\\n\\n".join(
        [
            f"Source: {doc.metadata.get('file_name', 'Unknown')}\\n{doc.page_content}"
            for doc in docs
        ]
    )

def ask_bot(question):

    guard_chain = (
        OOS_PROMPT
        | llm
        | StrOutputParser()
    )

    scope = guard_chain.invoke(
        {"question": question}
    )

    if scope.strip() != "IN_SCOPE":

        return (
            "I can only answer questions related to Zyro Dynamics HR policies.",
            []
        )

    docs = retriever.invoke(question)

    chain = (
        {
            "context": lambda _: format_docs(docs),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    return answer, docs

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

prompt = st.chat_input(
    "Ask your HR question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner(
        "Searching HR policies..."
    ):

        answer, docs = ask_bot(prompt)

    with st.chat_message("assistant"):

        st.markdown(answer)

        if docs:

            with st.expander("Sources"):

                source_files = list(
                    set(
                        doc.metadata.get(
                            "file_name",
                            "Unknown"
                        )
                        for doc in docs
                    )
                )

                for file in source_files:

                    st.write(f"• {file}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("app.py created successfully!")
