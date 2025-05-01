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

url = "https://71cec8e6-dff8-4b3e-84a0-46f9d71a9aed.europe-west3-0.gcp.cloud.qdrant.io"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.zVCndp5F7fuly8Xw7GkxnusyjYKWWIQsUjFpd9sJa9s"

qdrant = QdrantVectorStore.from_documents(
    docs,
    dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="annual_report_chunks",
)

def get_context(prompt: str) -> str:
    """
    """
    retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 7, 'fetch_k': 10})
    response = retriever.invoke(prompt)