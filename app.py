import streamlit as st
from ai_backend.parent_graph import multimodal_agentic_rag_graph, Chat_Input

# Configure Streamlit page
st.set_page_config(page_title="Multimodal Agentic RAG Chatbot", layout="wide")
st.title("Multimodal Agentic RAG Chatbot")

# Initialize chat history and processing state
if "history" not in st.session_state:
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Display chat history (always shows up-to-date history)
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["user"])
    
    with st.chat_message("assistant"):
        st.write(chat["assistant"])
    
    with st.expander("🔍 Show Context"):
        st.write(chat["context"])

# Show a placeholder for the assistant's response when processing
if st.session_state.processing:
    with st.chat_message("user"):
        st.write(st.session_state.current_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Process the input and get response
            chat_input = Chat_Input(
                prompt=st.session_state.current_input,
                assistant_id="",
                last_ai_message="",
                tranformed_queries=[],
                retrieved_docs=[],
                context=[]
            )
            
            response = multimodal_agentic_rag_graph.invoke(
                chat_input,
                config={"configurable": {"thread_id": st.session_state.get("thread_id", 1)}}
            )
            assistant_msg = response.get("last_ai_message", "")
            context = response.get("context", [])
            
            # Store in history
            st.session_state.history.append({
                "user": st.session_state.current_input,
                "assistant": assistant_msg,
                "context": context
            })
            
            # Reset processing flag
            st.session_state.processing = False
            
            # Rerun to update UI without the spinner
            st.rerun()

# Chat input (ChatGPT-like widget)
user_input = st.chat_input("Your message...", disabled=st.session_state.processing)

# On user submission, update state and trigger rerun
if user_input and not st.session_state.processing:
    st.session_state.current_input = user_input
    st.session_state.processing = True
    st.rerun()