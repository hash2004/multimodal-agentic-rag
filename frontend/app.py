import streamlit as st
from ai_backend.parent_graph import multimodal_agentic_rag_graph, Chat_Input


def main():
    # Configure page
    st.set_page_config(page_title="Multimodal Agentic RAG Chatbot", layout="wide")
    st.title("Multimodal Agentic RAG Chatbot")

    # Initialize chat history in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input form for user prompt
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Your message:", "")
        submitted = st.form_submit_button("Send")

    # On form submission, invoke backend and append to history
    if submitted and prompt:
        chat_input = Chat_Input(
            prompt=prompt,
            assistant_id="",
            last_ai_message="",
            tranformed_queries=[],
            retrieved_docs=[],
            context=[]
        )
        response = multimodal_agentic_rag_graph.invoke(chat_input)
        assistant_msg = response.get("last_ai_message", "")
        context = response.get("context", [])
        st.session_state.history.append({
            "user": prompt,
            "assistant": assistant_msg,
            "context": context
        })

    # Display chat history
    for idx, chat in enumerate(st.session_state.history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        # Expander to show context when clicked
        with st.expander("🔍 Show Context", expanded=False):
            st.write(chat["context"])


if __name__ == "__main__":
    main()
