import os
import logging
from operator import add
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
        collection_name=ANNUAL_REPORT_VECTOR_STORE,
    )
    logger.info(f"Successfully initialized Qdrant vector store with collection: {ANNUAL_REPORT_VECTOR_STORE}")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant vector store: {str(e)}")
    raise


class Annual_Input(MessagesState):
    """Input state for the Annual Report Assistant."""
    query: str
    context: List[Document]
    last_ai_message: str


class Annual_Output(TypedDict):
    """Output state for the Annual Report Assistant."""
    last_ai_message: str
    context: List[Document]


def get_context(state: Annual_Input) -> Dict[str, List[Document]]:
    """
    Retrieves relevant context documents based on the user query.
    
    Args:
        state: Current state containing the user query
        
    Returns:
        Dictionary with retrieved context documents
        
    Raises:
        Exception: If retrieval fails
    """
    logger.info(f"Getting context for query: {state['query']}")
    
    try:
        retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 5, 'fetch_k': 10})
        response = retriever.invoke(state['query'])
        
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


def get_answer(state: Annual_Input) -> Dict[str, str]:
    """
    Generates an answer based on the context and query.
    
    Args:
        state: Current state containing context and query
        
    Returns:
        Dictionary with the AI response
        
    Raises:
        Exception: If answer generation fails
    """
    logger.info(f"Generating answer for query: {state['query']}")
    
    try:
        chat_message = [
            SystemMessage(content=annual_report_assistant_prompt.format(
                context=state['context'],
                query=state['query']
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


# Initialize the state graph
a_chat_builder = StateGraph(input=Annual_Input, output=Annual_Output)

# Add nodes
a_chat_builder.add_node("get_context", get_context)
a_chat_builder.add_node("get_answer", get_answer)

# Add edges
a_chat_builder.add_edge(START, "get_context")
a_chat_builder.add_edge("get_context", "get_answer")
a_chat_builder.add_edge("get_answer", END)

# Initialize memory
memory = MemorySaver()

# Compile the graph
annual_report_assistant = a_chat_builder.compile(checkpointer=memory)
logger.info("Successfully compiled the Annual Report Assistant graph")

response = annual_report_assistant.invoke(
    {
        "query": "How many students graduated in 2023?",
        "context": [],
        "last_ai_message": ""
    },
    config={"configurable": {"thread_id": 2}}
)
print(response)