import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_openai import OpenAIEmbeddings

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
url = ""
api_key = ""

sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# first invocation
response = app.invoke(
    {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
    config={"configurable": {"thread_id": "1"}},
)

# extract and print only the latest AI reply
last_ai = next(
    (msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage)),
    None
)
if last_ai:
    print("Response:", last_ai.content)


# second invocation
response = app.invoke(
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config={"configurable": {"thread_id": "1"}},
)

last_ai = next(
    (msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage)),
    None
)
if last_ai:
    print("Response:", last_ai.content)

print("---------------------------------------------------------------")
#################
docs = []
qdrant_store = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="orignal_chunks",
)

retriever = qdrant_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50})
response = retriever.invoke("what were the stats of the illustrative balance sheet of assets?")
print("Response:", response)