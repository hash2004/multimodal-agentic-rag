import os
import logging
import random
from operator import add
from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_openai import OpenAIEmbeddings
from ai_backend.constants import *
from ai_backend.config import *
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from IPython.display import Image, display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
try:
    os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    logger.info("Environment variables set successfully")
except Exception as e:
    logger.error(f"Error setting environment variables: {e}")

# Initialize vector store connections
docs = []
url = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_URL")
api_key = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_API_KEY")

try:
    qdrant = QdrantVectorStore.from_documents(
        docs,
        dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name=FINANCIAL_VECTOR_STORE,
    )
    logger.info("Successfully initialized original chunks vector store")
except Exception as e:
    logger.error(f"Error initializing original chunks vector store: {e}")
    qdrant = None

try:
    qdrant_a = QdrantVectorStore.from_documents(
        docs,
        dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name=FINANCIAL_VECTOR_STORE_CONTEXULIZED,
    )
    logger.info("Successfully initialized contextualized chunks vector store")
except Exception as e:
    logger.error(f"Error initializing contextualized chunks vector store: {e}")
    qdrant_a = None


class Financial_Input(MessagesState):
    prompt: str
    last_ai_message: str
    tranformed_queries: Annotated[List[str], add]
    retrieved_docs: Annotated[List[Document], add]
    context: List[Document]
    
class Financial_Output(TypedDict):
    last_ai_message: str
    context: List[Document]

def transform_query_search_engine(state: Financial_Input) -> Dict[str, List[str]]:
    """
    Transform the user query for search engine optimization.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with transformed queries
    """
    logger.info("Transforming query for search engine")
    try:
        model = tranformed_query_model(query=state['prompt'])
        system_message = query_transformer_prompt.format(query=model.query)
        response = gemini_2_flash.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Provide the transformed query:")])
        
        transformed_query = response.content
        logger.info(f"Successfully transformed query for search engine: {transformed_query}")
        
        return {'tranformed_queries': [transformed_query]}
    except Exception as e:
        logger.error(f"Error transforming query for search engine: {e}")
        # Return default value to avoid pipeline failure
        return {'tranformed_queries': [state['prompt']]}


def transform_query_retrieval(state: Financial_Input) -> Dict[str, List[str]]:
    """
    Transform the user query for retrieval optimization.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with transformed queries
    """
    logger.info("Transforming query for retrieval")
    try:
        model = tranformed_query_model(query=state['prompt'])
        system_message = rewrite_query.format(query=model.query)
        response = gemini_2_flash.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Provide the transformed query:")])
        
        transformed_query = response.content
        logger.info(f"Successfully transformed query for retrieval: {transformed_query}")
        
        return {'tranformed_queries': [transformed_query]}
    except Exception as e:
        logger.error(f"Error transforming query for retrieval: {e}")
        # Return default value to avoid pipeline failure
        return {'tranformed_queries': [state['prompt']]}


def vector_o_search_orignal_mmr_0(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using MMR with transformed query at index 0.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using MMR with transformed query at index 0")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        transformed_query = state['tranformed_queries'][0]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_o_search_orignal_mmr_0: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_mmr_0: {e}")
        return {'retrieved_docs': [[]]}
    

def vector_c_search_orignal_mmr_0(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using MMR with transformed query at index 0.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using MMR with transformed query at index 0")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        transformed_query = state['tranformed_queries'][0]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_c_search_orignal_mmr_0: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_mmr_0: {e}")
        return {'retrieved_docs': [[]]}


def vector_o_search_orignal_similarity_0(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using similarity threshold with transformed query at index 0.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using similarity threshold with transformed query at index 0")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
        transformed_query = state['tranformed_queries'][0]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_o_search_orignal_similarity_0: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_similarity_0: {e}")
        return {'retrieved_docs': [[]]}


def vector_c_search_orignal_similarity_0(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using similarity threshold with transformed query at index 0.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using similarity threshold with transformed query at index 0")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
        transformed_query = state['tranformed_queries'][0]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_c_search_orignal_similarity_0: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_similarity_0: {e}")
        return {'retrieved_docs': [[]]}


def vector_o_search_orignal_mmr_1(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using MMR with transformed query at index 1.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using MMR with transformed query at index 1")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        transformed_query = state['tranformed_queries'][1]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_o_search_orignal_mmr_1: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_mmr_1: {e}")
        return {'retrieved_docs': [[]]}


def vector_c_search_orignal_mmr_1(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using MMR with transformed query at index 1.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using MMR with transformed query at index 1")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        transformed_query = state['tranformed_queries'][1]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_c_search_orignal_mmr_1: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_mmr_1: {e}")
        return {'retrieved_docs': [[]]}


def vector_o_search_orignal_similarity_1(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using similarity threshold with transformed query at index 1.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using similarity threshold with transformed query at index 1")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
        transformed_query = state['tranformed_queries'][1]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_o_search_orignal_similarity_1: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_similarity_1: {e}")
        return {'retrieved_docs': [[]]}


def vector_c_search_orignal_similarity_1(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using similarity threshold with transformed query at index 1.
    
    Args:
        state: The current state containing transformed queries
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using similarity threshold with transformed query at index 1")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
        transformed_query = state['tranformed_queries'][1]
        response = retriever.invoke(transformed_query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except IndexError as e:
        logger.error(f"Index error in vector_c_search_orignal_similarity_1: {e}")
        return {'retrieved_docs': [[]]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_similarity_1: {e}")
        return {'retrieved_docs': [[]]}


def vector_o_search_orignal_mmr_prompt(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using MMR with the original prompt.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using MMR with original prompt")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        query = state['prompt']
        response = retriever.invoke(query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_mmr_prompt: {e}")
        return {'retrieved_docs': [[]]}


def vector_c_search_orignal_mmr_prompt(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using MMR with the original prompt.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using MMR with original prompt")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
        query = state['prompt']
        response = retriever.invoke(query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_mmr_prompt: {e}")
        return {'retrieved_docs': [[]]}


def vector_o_search_orignal_similarity_prompt(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the original vector store using similarity threshold with the original prompt.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing original vector search using similarity threshold with original prompt")
    try:
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
        query = state['prompt']
        response = retriever.invoke(query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except Exception as e:
        logger.error(f"Error in vector_o_search_orignal_similarity_prompt: {e}")
        return {'retrieved_docs': [[]]}


def vector_c_search_orignal_similarity_prompt(state: Financial_Input) -> Dict[str, List[Document]]:
    """
    Search the contextualized vector store using similarity threshold with the original prompt.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with retrieved documents
    """
    logger.info("Performing contextualized vector search using similarity threshold with original prompt")
    try:
        if not qdrant_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
        query = state['prompt']
        response = retriever.invoke(query)
        
        logger.info(f"Successfully retrieved {len(response)} documents")
        return {'retrieved_docs': [response]}
    except Exception as e:
        logger.error(f"Error in vector_c_search_orignal_similarity_prompt: {e}")
        return {'retrieved_docs': [[]]}


def rerank_docs(state: Financial_Input) -> Dict[str, str]:
    """
    Rerank retrieved documents based on relevance to the query.
    
    Args:
        state: The current state containing the prompt
        
    Returns:
        Dictionary with reranked documents as formatted string
    """
    logger.info("Reranking documents")
    try:
        compressor = CohereRerank(
            model="rerank-english-v3.0", 
            top_n=4
        )
        
        if not qdrant:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        
        # Invoking the reranker
        compressed_docs = compression_retriever.invoke(state["prompt"])
        logger.info(f"Successfully reranked {len(compressed_docs)} documents")
        
        return {'context': compressed_docs}
    except Exception as e:
        logger.error(f"Error reranking documents: {e}")
        return {'context': []}


def chat_message(state: Financial_Input) -> Dict[str, str]:
    """
    Generate an AI response based on the retrieved documents and the prompt.
    
    Args:
        state: The current state containing retrieved documents and prompt
        
    Returns:
        Dictionary with AI message content
    """
    logger.info("Generating AI response")
    try:
        chat_message = [
            SystemMessage(content=financal_assistant_prompt.format(
                context=state['retrieved_docs'],
                query=state['prompt']
            )),
            HumanMessage(content="Provide the answer:"),
        ]
        
        response = gemini_2_flash.invoke(chat_message)
        logger.info("Successfully generated AI response")
        
        return {'last_ai_message': response.content}
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        return {'last_ai_message': "I'm sorry, I couldn't generate a response based on the available information."}


def gateway(state: Financial_Input):
    """
    Combine data from multiple sources (placeholder function).
    
    Args:
        state: The current state
        
    Returns:
        None
    """
    logger.info("gateway function called")
    try:
        return None
    except Exception as e:
        logger.error(f"Error in gateway function: {e}")
        return None


chat_builder = StateGraph(input=Financial_Input, output=Financial_Output)
chat_builder.add_node("transform_query_search_engine", transform_query_search_engine)
chat_builder.add_node("transform_query_retrieval", transform_query_retrieval)

chat_builder.add_node("vector_o_search_orignal_mmr_0", vector_o_search_orignal_mmr_0)
chat_builder.add_node("vector_c_search_orignal_mmr_0", vector_c_search_orignal_mmr_0)
chat_builder.add_node("vector_o_search_orignal_similarity_0", vector_o_search_orignal_similarity_0)
chat_builder.add_node("vector_c_search_orignal_similarity_0", vector_c_search_orignal_similarity_0)

chat_builder.add_node("vector_o_search_orignal_mmr_1", vector_o_search_orignal_mmr_1)
chat_builder.add_node("vector_c_search_orignal_mmr_1", vector_c_search_orignal_mmr_1)
chat_builder.add_node("vector_o_search_orignal_similarity_1", vector_o_search_orignal_similarity_1)
chat_builder.add_node("vector_c_search_orignal_similarity_1", vector_c_search_orignal_similarity_1)

chat_builder.add_node("vector_o_search_orignal_mmr_prompt", vector_o_search_orignal_mmr_prompt)
chat_builder.add_node("vector_c_search_orignal_mmr_prompt", vector_c_search_orignal_mmr_prompt)
chat_builder.add_node("vector_o_search_orignal_similarity_prompt", vector_o_search_orignal_similarity_prompt)
chat_builder.add_node("vector_c_search_orignal_similarity_prompt", vector_c_search_orignal_similarity_prompt)

chat_builder.add_node("rerank_docs", rerank_docs)
chat_builder.add_node("chat_message", chat_message)

chat_builder.add_node("gateway", gateway)

chat_builder.add_edge(START, "transform_query_search_engine")
chat_builder.add_edge(START, "transform_query_retrieval")

chat_builder.add_edge("transform_query_retrieval", "gateway")
chat_builder.add_edge("transform_query_search_engine", "gateway")

chat_builder.add_edge("gateway", "vector_o_search_orignal_mmr_0")
chat_builder.add_edge("gateway", "vector_c_search_orignal_mmr_0")
chat_builder.add_edge("gateway", "vector_o_search_orignal_similarity_0")
chat_builder.add_edge("gateway", "vector_c_search_orignal_similarity_0")

chat_builder.add_edge("gateway", "vector_o_search_orignal_mmr_1")
chat_builder.add_edge("gateway", "vector_c_search_orignal_mmr_1")
chat_builder.add_edge("gateway", "vector_o_search_orignal_similarity_1")
chat_builder.add_edge("gateway", "vector_c_search_orignal_similarity_1")

chat_builder.add_edge("gateway", "vector_o_search_orignal_mmr_prompt")
chat_builder.add_edge("gateway", "vector_c_search_orignal_mmr_prompt")
chat_builder.add_edge("gateway", "vector_o_search_orignal_similarity_prompt")
chat_builder.add_edge("gateway", "vector_c_search_orignal_similarity_prompt")

chat_builder.add_edge("vector_o_search_orignal_mmr_0", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_mmr_0", "rerank_docs")
chat_builder.add_edge("vector_o_search_orignal_similarity_0", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_similarity_0", "rerank_docs")

chat_builder.add_edge("vector_o_search_orignal_mmr_1", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_mmr_1", "rerank_docs")
chat_builder.add_edge("vector_o_search_orignal_similarity_1", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_similarity_1", "rerank_docs")

chat_builder.add_edge("vector_o_search_orignal_mmr_prompt", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_mmr_prompt", "rerank_docs")
chat_builder.add_edge("vector_o_search_orignal_similarity_prompt", "rerank_docs")
chat_builder.add_edge("vector_c_search_orignal_similarity_prompt", "rerank_docs")

chat_builder.add_edge("rerank_docs", "chat_message")
chat_builder.add_edge("chat_message", END)

memory = MemorySaver()

financial_assistant = chat_builder.compile(checkpointer=memory)

"""
response = financial_assistant.invoke(
    {
        "prompt": "What is the revenue for the year 2023?",
        "last_ai_message": "",
        "tranformed_queries": [],
        "retrieved_docs": [],
        "context": []
    },
    config={"configurable": {"thread_id": 1}},
)

print(response)
"""