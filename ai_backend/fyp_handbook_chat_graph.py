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
from langchain_groq import ChatGroq

docs = []

url = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_URL")
api_key = os.getenv("QDRANT_CLOUD_MULTIMODAL_AGENTIC_RAG_API_KEY")

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

class AChatInput(MessagesState):
    query: str
    retrieved_docs: Annotated[List[Document], add]
    last_ai_message: StopAsyncIteration

class AChatOutput(TypedDict):
    last_ai_message: str
    retrieved_docs: Annotated[List[Document], add]
    

def get_context(state: AChatInput) -> str:
    """
    """
    retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 3, 'fetch_k': 10})
    response = retriever.invoke(state['query'])
    
    return {
        "retrieved_docs": [response]
    }
    
    
def get_answer(state: AChatInput) -> Dict[str, str]:
    chat_message = [
        SystemMessage(content=fyp_handbook_assistant_prompt.format(
            context=state['retrieved_docs'],
            query=state['query']
        )),
        HumanMessage(content="Provide the answer:"),
    ]
    
    response = gemini_2_flash.invoke(chat_message)
    
    return {'last_ai_message': response.content}
    

f_chat_builder = StateGraph(input=AChatInput, output=AChatOutput)

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
        "query": "What is the role of FYP coordinator?",
        "retrieved_docs": [],
        "last_ai_message": ""
    },
    config={"configurable": {"thread_id": 3}}
)

print(response)
"""