import streamlit as st

st.set_page_config(page_title="Zyro Dynamics HR Help Desk")

st.title("🏢 Zyro Dynamics HR Help Desk")
st.write("Ask HR-related questions based on company policies.")

REFUSAL_MESSAGE = "I can only answer questions related to Zyro Dynamics HR policies."

def chatbot(question):
    hr_keywords = [
        "leave","policy","employee","salary","payroll",
        "benefits","vacation","travel","expense",
        "conduct","security","performance",
        "onboarding","resignation","termination","hr"
    ]

    if not any(word in question.lower() for word in hr_keywords):
        return REFUSAL_MESSAGE

    return "This is where your RAG response will be returned."

question = st.text_input("Ask your question")

if st.button("Submit") and question:
    answer = chatbot(question)
    st.write(answer)