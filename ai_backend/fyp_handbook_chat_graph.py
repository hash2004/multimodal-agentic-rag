import os
import logging
from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from ai_backend.constants import *
from ai_backend.config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

docs = []

# Get environment variables with proper error handling
try:
    url = os.environ.get("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_URL")
    api_key = os.environ.get("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_API_KEY")
    
    if not url or not api_key:
        raise ValueError("Missing required environment variables: QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_URL or QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_API_KEY")
        
    logger.info("Successfully loaded Qdrant environment variables")
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}")
    raise

# Initialize Qdrant vector store with proper error handling
try:
    qdrant = QdrantVectorStore.from_documents(
        docs,
        dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name=FYP_HANDBOOK_VECTOR_STORE_CONTEXULIZED,
    )
    logger.info(f"Successfully initialized Qdrant vector store with collection: {FYP_HANDBOOK_VECTOR_STORE_CONTEXULIZED}")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant vector store: {str(e)}")
    raise


class FYP_Input(MessagesState):
    """Input state for the FYP Handbook Assistant."""
    prompt: str
    context: List[Document]
    last_ai_message: str


class FYP_Output(TypedDict):
    """Output state for the FYP Handbook Assistant."""
    last_ai_message: str
    context: List[Document]


def get_context(state: FYP_Input) -> Dict[str, List[Document]]:
    """
    Retrieves relevant context documents from the vector store based on the user prompt.
    
    Args:
        state: Current state containing the user prompt
        
    Returns:
        Dictionary with retrieved context documents
        
    Raises:
        Exception: If retrieval fails
    """
    logger.info(f"Getting context for prompt: {state['prompt']}")
    
    try:
        retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 5, 'fetch_k': 10})
        response = retriever.invoke(state['prompt'])
        
        logger.info(f"Successfully retrieved {len(response)} context documents")
        return {
            "context": [response]
        }
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        # Return empty context to prevent complete failure
        return {
            "context": []
        }


def get_answer(state: FYP_Input) -> Dict[str, str]:
    """
    Generates an answer based on the context and prompt using the Gemini model.
    
    Args:
        state: Current state containing context and prompt
        
    Returns:
        Dictionary with the AI response
        
    Raises:
        Exception: If answer generation fails
    """
    logger.info(f"Generating answer for prompt: {state['prompt']}")
    
    try:
        chat_message = [
            SystemMessage(content=fyp_handbook_assistant_prompt.format(
                context=state['context'],
                query=state['prompt']
            )),
            HumanMessage(content="Provide the answer:"),
        ]
        
        response = gemini_2_flash.invoke(chat_message)
        logger.info("Successfully generated answer")
        
        return {'last_ai_message': response.content}
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        # Return a graceful failure message
        return {'last_ai_message': "I'm sorry, I encountered an error while processing your request. Please try again."}


f_chat_builder = StateGraph(input=FYP_Input, output=FYP_Output)

f_chat_builder.add_node("get_context", get_context)
f_chat_builder.add_node("get_answer", get_answer)

f_chat_builder.add_edge(START, "get_context")
f_chat_builder.add_edge("get_context", "get_answer")
f_chat_builder.add_edge("get_answer", END)

memory = MemorySaver()

fyp_handbook_assistant = f_chat_builder.compile(checkpointer=memory)

"""
response = fyp_handbook_assistant.invoke(
    {
        "prompt": "What is the role of FYP coordinator?",
        "context": [],
        "last_ai_message": ""
    },
    config={"configurable": {"thread_id": 3}}
)

print(response)
"""