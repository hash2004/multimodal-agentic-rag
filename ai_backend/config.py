from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os

os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")