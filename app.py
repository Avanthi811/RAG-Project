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

# ── PAGE CONFIGURATION ───────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── ENV CONFIGURATION & CONSTANTS ───────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

REFUSAL_MESSAGE = "I am sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. Please contact your HR team for other queries."

# Strict RAG prompt ensuring compliance with the source text
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a precise HR assistant for Zyro Dynamics. 
Answer the employee question using ONLY the context provided below. 
Be highly accurate, professional, and concise. 
If the context does not contain the answer, state exactly: "I am sorry, I can only answer HR-related questions based on Zyro Dynamics policy documents. Please contact your HR team for other queries."

Context:
{context}

Question: {question}

Answer:""")

# Secondary guardrail check prompt
OOS_PROMPT = ChatPromptTemplate.from_template("""You are a security classifier. 
Determine if this question is related to company internal operational HR policies, employee benefits, leaves, or company procedures.
Reply with ONLY 'yes' or 'no'.

Question: {question}
""")

# ── RAG PIPELINE INITIALIZATION ─────────────────────────────
@st.cache_resource(show_spinner="Loading HR documents and building knowledge base...")
def build_pipeline():
    # Fetch secrets directly from Streamlit's environment configuration
    os.environ["GROQ_API_KEY"]      = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

    # Target directory for local setup and deployment
    pdf_files = glob.glob("hr_docs/*.pdf")
    documents = []
    
    for pdf_path in sorted(pdf_files):
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Skipping unreadable file {os.path.basename(pdf_path)}: {e}")

    if not documents:
        st.error("No PDF documents found in 'hr_docs/' folder. Please place policy PDFs there.")
        st.stop()

    # Highly optimized splitter strategy to prevent losing context across sentence breaks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks   = splitter.split_documents(documents)

    # Embed chunks using an production-standard lightweight model
    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Maximal Marginal Relevance retrieval pattern ensures contextual diversity
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )

    # Use llama-3.1-8b-instant or llama-3.3-70b-versatile based on subscription rules
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return retriever, llm

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ── EXECUTION PIPELINES ─────────────────────────────────────
@traceable(name="rag_chain")
def rag_chain(question, retriever, llm):
    docs     = retriever.invoke(question)
    context  = format_docs(docs)
    prompt   = RAG_PROMPT.format(context=context, question=question)
    response
