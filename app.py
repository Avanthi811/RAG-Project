import streamlit as st

# We assume ask_bot is available from your backend code
# (i.e., the same logic as in the notebook).

st.title("Zyro Dynamics HR Help Desk")

st.write("Ask any question about Zyro Dynamics HR policies.")

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
