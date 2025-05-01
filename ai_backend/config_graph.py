import os
import logging
import random
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_openai import OpenAIEmbeddings

from ai_backend.constants import (
    FINANCIAL_BASIC,
    FINANCIAL_ADVANCED
)
from ai_backend.config import dense_embeddings, sparse_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('langgraph_config.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
QDRANT_URL = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_URL")
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_API_KEY")

# Validate environment variables
if not QDRANT_URL or not QDRANT_API_KEY:
    logger.error("Missing required environment variables: QDRANT_URL or QDRANT_API_KEY")
    raise EnvironmentError("Missing required Qdrant configuration environment variables")


class Config(TypedDict):
    """Input configuration for the graph."""
    document: str
    thread_id: int
    qdrant_orignal: Optional[QdrantVectorStore]
    qdrant_advanced: Optional[QdrantVectorStore]


class ConfigOutput(TypedDict):
    """Output configuration returned by the graph."""
    thread_id: int
    qdrant_orignal: QdrantVectorStore
    qdrant_advanced: QdrantVectorStore


def configure_thread(state: Config) -> Dict[str, str]:
    """
    Generate a unique thread ID for the conversation.
    
    Args:
        state: The current configuration state
        
    Returns:
        Dictionary containing the generated thread ID
    """
    try:
        thread_id = random.randint(1, 100000)
        logger.info(f"Created new thread ID: {thread_id}")
        return {"thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error generating thread ID: {e}")
        raise


def configure_vector_store(state: Config) -> Dict[str, QdrantVectorStore]:
    """
    Configure and initialize vector stores based on document type.
    
    Args:
        state: The current configuration state including document type
        
    Returns:
        Dictionary containing configured vector stores
    """
    document_type = state.get("document", "").lower()
    logger.info(f"Configuring vector stores for document type: {document_type}")
    
    docs: List[Document] = []  # Placeholder for actual documents
    
    # Default collection names
    basic_collection = None
    advanced_collection = None
    
    try:
        # Set collection names based on document type
        if document_type == "financial":
            basic_collection = FINANCIAL_BASIC
            advanced_collection = FINANCIAL_ADVANCED
        elif document_type == "legal":
           pass
        else:
            logger.error(f"Unsupported document type: {document_type}")
            raise ValueError(f"Unsupported document type: {document_type}")
        
        logger.info(f"Using collections: {basic_collection} and {advanced_collection}")
        
        # Initialize vector stores
        qdrant_original = create_vector_store(docs, basic_collection)
        qdrant_advanced = create_vector_store(docs, advanced_collection)
        
        return {
            "qdrant_orignal": qdrant_original,
            "qdrant_advanced": qdrant_advanced,
        }
        
    except Exception as e:
        logger.error(f"Error configuring vector stores: {e}")
        raise


def create_vector_store(docs: List[Document], collection_name: str) -> QdrantVectorStore:
    """
    Create a Qdrant vector store with the specified configuration.
    
    Args:
        docs: Documents to initialize the vector store with
        collection_name: Name of the Qdrant collection
        
    Returns:
        Configured QdrantVectorStore instance
    """
    try:
        logger.info(f"Creating vector store for collection: {collection_name}")
        
        vector_store = QdrantVectorStore.from_documents(
            docs,
            dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
        )
        
        logger.info(f"Successfully created vector store for {collection_name}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to create vector store for {collection_name}: {e}")
        raise


def build_config_graph() -> StateGraph:
    """
    Build and return the configuration graph.
    
    Returns:
        Compiled StateGraph for configuration
    """
    try:
        logger.info("Building configuration graph")
        
        # Initialize the state graph
        config_builder = StateGraph(input=Config, output=ConfigOutput)
        
        # Add nodes
        config_builder.add_node("configure_thread", configure_thread)
        config_builder.add_node("configure_vector_store", configure_vector_store)
        
        # Connect nodes
        config_builder.add_edge(START, "configure_thread")
        config_builder.add_edge("configure_thread", "configure_vector_store")
        config_builder.add_edge("configure_vector_store", END)
        
        # Compile the graph
        config_graph = config_builder.compile()
        
        logger.info("Configuration graph successfully built")
        return config_graph
        
    except Exception as e:
        logger.error(f"Error building configuration graph: {e}")
        raise


# Create the configuration graph
config_graph = build_config_graph()


def run_config(document_type: str) -> ConfigOutput:
    """
    Run the configuration graph with the specified document type.
    
    Args:
        document_type: Type of documents to configure (e.g., "financial", "legal")
        
    Returns:
        Configuration output including thread ID and vector stores
    """
    try:
        logger.info(f"Running configuration for document type: {document_type}")
        
        response = config_graph.invoke(
            {
                "document": document_type,
                "thread_id": "",
                "qdrant_orignal": None,
                "qdrant_advanced": None,
            }
        )
        
        logger.info(f"Configuration completed successfully for {document_type}")
        return response
        
    except ValueError as e:
        logger.error(f"Value error during configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during configuration: {e}")
        raise


"""
try:
    response = run_config("financial")
    print("---------------------------------------------------------------")
    print("Response:", response)
    print(f"Thread ID: {response['thread_id']}")
    print(f"Vector stores configured: {bool(response['qdrant_orignal'])}, {bool(response['qdrant_advanced'])}")
except Exception as e:
    logger.critical(f"Failed to run configuration: {e}")
"""