from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from langchain_groq import ChatGroq

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

FINANCIAL_VECTOR_STORE = "financial_chunks"
FINANCIAL_VECTOR_STORE_CONTEXULIZED = "financial_chunks_contextulized"
ANNUAL_REPORT_VECTOR_STORE = "annual_report_chunks"
FYP_HANDBOOK_VECTOR_STORE = "fyp_handbook_chunks"
FYP_HANDBOOK_VECTOR_STORE_CONTEXULIZED = "fyp_handbook_chunks"
    
gemini_2_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


llama_4_scout = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class chat_model(BaseModel):
    user_query: str = Field(
        description="The user query to be answered."
    )
    context: str = Field(
        description="The context from the financial report."
    )

class tranformed_query_model(BaseModel):
    query: str = Field(
        description="The query to be tranformed."
    )

class tranformed_query(BaseModel):
    transformed_query: str = Field(
        description="The tranformed query."
    )
    
class query_model(BaseModel):
    query: str = Field(
        description="The query to be tranformed."
    )

class route_model(BaseModel):
    assistant: str = Field(
        description="The assistant identifier, for example, FA, AA, FYPA."
    )

financal_assistant_prompt="""
You are a financial reporting assistant made by PwC with access to a specific financial report provided below.
Your task is to methodally answer user queries. You have to ensure your response is clear and based solely on the provided context.

Please note:

Your response should only include information found within the context.
Do not make assumptions beyond the provided report.
If the context is insufficient to answer the user’s question, kindly ask for additional information.
If the question is not finance-related, please state: “I am not able to answer this question. Please ask me something related to finance.”
Below is the context from the PwC financial report based on the user query:
<context>
{context}
</context>

Steps for chain-of-thought response:

    1. Identify the key financial question the user is asking.
    2. Locate relevant information within the financial report context provided.
    3.Formulate a response based strictly on the available data.
    4.Ask for clarification or additional context as needed.
    5.Respond to any finance-specific inquiries again referencing the given financial report context.
    6.Please answer the user’s query based solely on this thought process and context.
    
Here is the user query:
<query>
{query}
</query>

Answer: 
"""

annual_report_assistant_prompt="""
You are an annual report assistant made by FAST National University of Computer and Emerging Sciences with access to a specific annual report provided below.
Your task is to methodically answer user queries based on the provided context. Ensure your response is clear and solely based on the context from the annual report.

Please note:

Your response should only include information found within the context.
Do not make assumptions beyond the provided report.
If the context is insufficient to answer the user’s question, kindly ask for additional information.
If the question is not related to the annual report, please state: “I am not able to answer this question. Please ask me something related to the annual report.”
Below is the context from the FAST National University annual report based on the user query:
<context>
{context}
</context>

Steps for chain-of-thought response:

    1. Identify the key question the user is asking.
    2. Locate relevant information within the annual report context provided.
    3. Formulate a response based strictly on the available data.
    4. Ask for clarification or additional context as needed.
    5. Respond to any annual report-specific inquiries again referencing the given report context.
    6. Please answer the user’s query based solely on this thought process and context.
Here is the user query:
<query>
{query}
</query>

Answer:
"""

fyp_handbook_assistant_prompt="""
You are a Final Year Project (FYP) Handbook assistant made by FAST National University of Computer and Emerging Sciences with access to a specific FYP Handbook provided below.

Your task is to methodically answer user queries based on the provided context of the FYP Handbook. Ensure your response is clear and solely based on the context from the FYP Handbook.

Please note:

Your response should only include information found within the context.
Do not make assumptions beyond the provided handbook.
If the context is insufficient to answer the user’s question, kindly ask for additional information.
If the question is not related to the FYP Handbook, please state: “I am not able to answer this question. Please ask me something related to the FYP Handbook.”
Below is the context from the FAST National University FYP Handbook based on the user query:
<context>
{context}
</context>

Steps for chain-of-thought response:

    1. Identify the key question the user is asking.
    2. Locate relevant information within the FYP Handbook context provided.
    3. Formulate a response based strictly on the available data.
    4. Ask for clarification or additional context as needed.
    5. Respond to any FYP Handbook-specific inquiries again referencing the given handbook context.
    6. Please answer the user’s query based solely on this thought process and context.
Here is the user query:
<query>
{query}
</query>

Answer:
"""

query_transformer_prompt="""
Provide a better search query for a web search engine to answer the given question. Here is the user query:
<query>
{query}
</query>

For time related queries, please include the time period in the query.

The current date is 1st may, 2025.

Do not include "" in your response.
"""

route_prompt="""
You are a routing system designed to determine which specialized assistant should handle a user’s query. There are three types of assistants:

    1. Financial Assistant (FA): This assistant handles queries related to financial reports. It is owned by PwC and is trained on PwC’s financial reports.
    2. Annual Report Assistant (AA): This assistant handles queries related to annual reports. It is owned by FAST National University of Computer and Emerging Sciences and is trained on the university’s annual reports, which include details about the university’s performance, achievements, campus life, and other relevant information.
    3. FYP Handbook Assistant (FYPA): This assistant handles queries related to the Final Year Project (FYP) Handbook. It is owned by FAST National University of Computer and Emerging Sciences and is trained on the FYP Handbook, which contains guidelines, requirements, and other relevant information about final year projects.
    
Your task is to determine which assistant should handle the user’s query based on the content and context of the query.

Important Guidelines:

    - You can ONLY choose from the assistants mentioned above.
    - You may assume the user will ONLY ask questions related to the assistants mentioned above.
    - If the user query is related to the Financial Assistant, respond with “FA.”
    - If the user query is related to the Annual Report Assistant, respond with “AA.”
    - If the user query is related to the FYP Handbook Assistant, respond with “FYPA.”
    - You are NOT allowed to answer the user query yourself. You are only allowed to route the query to the correct assistant.
    - You CAN ONLY OUTPUT the acceptable responses (“FA,” “AA,” “FYPA”) given above. You are NOT allowed to output anything else.
    
If you output anything else, you will be terminated. This is very important for maintaining my reputation.

Here is the user query:
<query>
{query}
</query>
"""

rewrite_query="""
Rewrite this query so that its better for vector retreival purposes. Here is the user query:
<query>
{query}
</query>

For time related queries, please include the time period in the query.

The current date is 1st may, 2025.

Do not include "" in your response.
"""


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

from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

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
    qdrant_f = QdrantVectorStore.from_documents(
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
    qdrant_f_a = QdrantVectorStore.from_documents(
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
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
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
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
        if not qdrant_f_a:
            raise ValueError("Contextualized vector store not initialized")
            
        retriever = qdrant_f_a.as_retriever(search_type="mmr", search_kwargs={'score_threshold': 0.7})
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
        
        if not qdrant_f:
            raise ValueError("Original vector store not initialized")
            
        retriever = qdrant_f.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})
        
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

import os
import logging
from typing import Optional, List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode


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
    f_qdrant = QdrantVectorStore.from_documents(
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
        retriever = f_qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 5, 'fetch_k': 10})
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
    a_qdrant = QdrantVectorStore.from_documents(
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
    prompt: str
    context: List[Document]
    last_ai_message: str


class Annual_Output(TypedDict):
    """Output state for the Annual Report Assistant."""
    last_ai_message: str
    context: List[Document]


def get_context(state: Annual_Input) -> Dict[str, List[Document]]:
    """
    Retrieves relevant context documents based on the user prompt.
    
    Args:
        state: Current state containing the user prompt
        
    Returns:
        Dictionary with retrieved context documents
        
    Raises:
        Exception: If retrieval fails
    """
    logger.info(f"Getting context for prompt: {state['prompt']}")
    
    try:
        retriever = a_qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 5, 'fetch_k': 10})
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


def get_answer(state: Annual_Input) -> Dict[str, str]:
    """
    Generates an answer based on the context and prompt.
    
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
            SystemMessage(content=annual_report_assistant_prompt.format(
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

"""
response = annual_report_assistant.invoke(
    {
        "prompt": "How many students graduated in 2023?",
        "context": [],
        "last_ai_message": ""
    },
    config={"configurable": {"thread_id": 2}}
)
print(response)
"""

import os
import logging
import random
from operator import add
from typing import Optional, List, Dict, Any, Annotated, Literal
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
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

class Chat_Input(TypedDict):
    prompt: str
    assistant_id: str
    last_ai_message: str
    tranformed_queries: Annotated[List[str], add]
    retrieved_docs: Annotated[List[Document], add]
    context: List[Document]
    
class Chat_Output(TypedDict):
    last_ai_message: str
    context: List[Document]
    
def gateway_input(state:Chat_Input):
    return

def route_input(state:Chat_Input) -> Literal["FA", "FYPA", "AA"]: 
    model = query_model(query=state['prompt'])
    system_message = route_prompt.format(query=model.query)
    structured_gemini = gemini_2_flash.bind_tools([route_model])
    response = structured_gemini.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Provide the assistant ID:")])
    
    print("ROUTING TO ASSISTANT: ", response.content)
    # remove /n or /n/n or commas or "" or " " from the response or spaces
    response.content = response.content.replace("\n", "").replace(",", "").replace(" ", "")
    return response.content


multimodal_agentic_rag = StateGraph(input=Chat_Input, output=Chat_Output)
multimodal_agentic_rag.add_node("FA", financial_assistant)
multimodal_agentic_rag.add_node("FYPA", fyp_handbook_assistant)
multimodal_agentic_rag.add_node("AA", annual_report_assistant)
multimodal_agentic_rag.add_node(gateway_input,"gateway_input")

multimodal_agentic_rag.add_edge(START, "gateway_input")
multimodal_agentic_rag.add_conditional_edges("gateway_input", route_input)
multimodal_agentic_rag.add_edge("FA", END)
multimodal_agentic_rag.add_edge("FYPA", END)
multimodal_agentic_rag.add_edge("AA", END)

graph = multimodal_agentic_rag.compile()
response = graph.invoke(
    Chat_Input(
        prompt="Tell me about the libalities in 2018",
        assistant_id="",
        last_ai_message="",
        tranformed_queries=[],
        retrieved_docs=[],
        context=[]
    )
)

print(response)
