import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼")

CORPUS_PATH = "/mount/src/zyro-dynamics-hr-corpus"  # change after upload if needed

@st.cache_resource
def build_bot():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    rag_prompt = ChatPromptTemplate.from_template(
        \"\"\"You are an HR assistant for Zyro Dynamics.
Use ONLY the context from the HR policy documents to answer the employee's question.
If the answer is not in the documents, say you don't know and suggest contacting HR.

Context:
{context}

Question:
{question}

Answer clearly and concisely in 3–5 sentences.\"\"\"
    )

    oos_prompt = ChatPromptTemplate.from_template(
        \"\"\"You are an HR guardrail classifier for Zyro Dynamics.
Decide if the user question is about Zyro HR policies, benefits, leave, payroll, attendance,
work-from-home, working hours, performance appraisal, or similar internal HR topics.

Question:
{question}

If the question is about Zyro HR policies, answer exactly: IN_SCOPE.
If it is not about Zyro HR policies, answer exactly: OUT_OF_SCOPE.\"\"\"
    )

    refusal_message = (
        "I am designed only to answer questions about Zyro Dynamics HR policies. "
        "Your question appears to be outside this scope. "
        "Please contact the relevant support team for non-HR queries."
    )

    def format_docs(docs):
        return "\\n\\n".join(doc.page_content for doc in docs)

    def rag_chain(question: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        prompt = rag_prompt.format(context=context, question=question)
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def ask_bot(question: str):
        cls_prompt = oos_prompt.format(question=question)
        cls_resp = llm.invoke(cls_prompt)
        label = cls_resp.content.strip().upper()

        if label == "OUT_OF_SCOPE":
            return {"answer": refusal_message, "scope": "out_of_scope"}

        answer_text = rag_chain(question)
        return {"answer": answer_text, "scope": "in_scope"}

    return ask_bot

st.title("Zyro Dynamics HR Help Desk")
st.write("Ask any question about Zyro Dynamics HR policies.")

try:
    ask_bot = build_bot()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

user_question = st.text_input("Your question")

if "chat_history" not in st.session_state:
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
