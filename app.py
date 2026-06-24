import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

# ─────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# LangSmith Tracing
# ─────────────────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
REFUSAL_MESSAGE = (
    "I am sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics policy documents. Please contact your HR team "
    "for other queries."
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise HR assistant for Zyro Dynamics.

Answer the employee question using ONLY the context provided below.

Rules:
1. Do not use outside knowledge.
2. Be professional and concise.
3. If the answer is not found in the context, respond exactly with:
"I am sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. Please contact your HR team for other queries."

Context:
{context}

Question:
{question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a security classifier.

Determine whether the question is related to:
- HR policies
- Leave policies
- Employee benefits
- Company procedures
- Internal employee guidelines

Reply with ONLY:
yes
or
no

Question:
{question}
""")

# ─────────────────────────────────────────────────────────────
# Build RAG Pipeline
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR documents and building knowledge base...")
def build_pipeline():

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

    pdf_files = glob.glob("hr_docs/*.pdf")

    if not pdf_files:
        st.error("No PDF files found in the 'hr_docs' folder.")
        st.stop()

    documents = []

    for pdf_path in sorted(pdf_files):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        st.error("No document chunks were created.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm


# ─────────────────────────────────────────────────────────────
# Helper Function
# ─────────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────
# RAG Chain
# ─────────────────────────────────────────────────────────────
@traceable(name="rag_chain")
def rag_chain(question, retriever, llm):

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    context = format_docs(docs)

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    sources = list({
        os.path.basename(
            doc.metadata.get("source", "Unknown")
        )
        for doc in docs
    })

    return {
        "answer": response.content.strip(),
        "sources": sources
    }


# ─────────────────────────────────────────────────────────────
# Main Bot Logic
# ─────────────────────────────────────────────────────────────
@traceable(name="ask_bot")
def ask_bot(question, retriever, llm):

    check_prompt = OOS_PROMPT.format(
        question=question
    )

    classification = llm.invoke(
        check_prompt
    ).content.strip().lower()

    if classification == "yes":
        return rag_chain(
            question,
            retriever,
            llm
        )

    return {
        "answer": REFUSAL_MESSAGE,
        "sources": []
    }


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question — powered by RAG")

retriever, llm = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask an HR question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):
            result = ask_bot(
                question,
                retriever,
                llm
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.write(f"• {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
