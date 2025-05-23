import os
import logging
import base64
from operator import add
from IPython.display import Image, display
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
from ai_backend.constants import *
from ai_backend.config import *
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from ai_backend.annual_report_chat_graph import annual_report_assistant
from ai_backend.financial_chat_graph import financial_assistant
from ai_backend.fyp_handbook_chat_graph import fyp_handbook_assistant
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Chat_Input(MessagesState):
    """
    Input state for the chat processing pipeline.
    
    Attributes:
        prompt: The user's input prompt
        assistant_id: Identifier for the assistant
        last_ai_message: The most recent message from the AI
        tranformed_queries: List of transformed queries derived from the prompt
        retrieved_docs: List of documents retrieved based on the queries
        context: List of contextual documents
    """
    prompt: str
    image_path: str
    assistant_id: str
    last_ai_message: str
    tranformed_queries: Annotated[List[str], add]
    retrieved_docs: Annotated[List[Document], add]
    context: List[Document]
    
class Chat_Output(TypedDict):
    """
    Output state from the chat processing pipeline.
    
    Attributes:
        last_ai_message: The most recent message from the AI
        context: List of contextual documents
    """
    last_ai_message: str
    context: List[Document]
    
def gateway_input(state: Chat_Input):
    """
    Process multimodal input by extracting image content using Gemini and combining it with text prompt.
    
    Args:
        state: The input state containing prompt and image path
    
    Returns:
        Dict with processed prompt including image description
    """
    try:
        # Validate image path
        if not state.get('image_path'):
            logger.warning("No image path provided in state")
            return {"prompt": state['prompt']}
            
        if not os.path.isfile(state['image_path']):
            logger.error(f"Invalid image path: {state['image_path']}")
            return {"prompt": state['prompt']}
            
        logger.info(f"Processing image from path: {state['image_path']}")
        
        # Read and encode image
        try:
            with open(state['image_path'], "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                logger.debug("Successfully encoded image")
        except Exception as e:
            logger.error(f"Error reading or encoding image: {str(e)}")
            return {"prompt": state['prompt']}
        
        # Create message with image for Gemini
        message_local = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the image in full detail."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
            ]
        )
        
        # Get image description from Gemini
        try:
            logger.debug("Invoking Gemini for image description")
            result = gemini_2_flash.invoke([message_local])
            final_result = result.content
            logger.info("Successfully received image description from Gemini")
        except Exception as e:
            logger.error(f"Error invoking Gemini for image description: {str(e)}")
            return {"prompt": state['prompt']}
        
        # Combine original prompt with image description
        query = state['prompt'] + "\nThe user also inserted an image, the content of the image is: \n" + final_result
        logger.debug(f"Combined prompt created with image description")
        
        return {
            "prompt": query,
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in gateway_input: {str(e)}")
        # Return original prompt if any unexpected error occurs
        return {"prompt": state.get('prompt', '')}

def route_input(state: Chat_Input) -> Literal["FA", "FYPA", "AA"]:
    """
    Route the input to the appropriate assistant type.
    
    Args:
        state: The input state containing prompt and context information
    
    Returns:
        String literal indicating which assistant type to use:
        "FA" (Factual Assistant), "FYPA" (For Your Personal Assistant), or "AA" (Analytical Assistant)
    """
    try:
        logger.info(f"Routing input for prompt: {state['prompt'][:50]}...")
        
        try:
            model = query_model(query=state['prompt'])
            logger.debug("Successfully queried model")
        except Exception as e:
            logger.error(f"Error querying model: {str(e)}")
            raise
        
        try:
            system_message = route_prompt.format(query=model.query)
            logger.debug("Created system message for routing")
        except Exception as e:
            logger.error(f"Error formatting route prompt: {str(e)}")
            raise
        
        try:
            structured_gemini = gemini_2_flash.bind_tools([route_model])
            logger.debug("Bound tools to Gemini model")
        except Exception as e:
            logger.error(f"Error binding tools to Gemini: {str(e)}")
            raise
        
        try:
            response = structured_gemini.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Provide the assistant ID:")])
            logger.debug("Successfully invoked structured Gemini")
        except Exception as e:
            logger.error(f"Error invoking structured Gemini: {str(e)}")
            raise
        
        response.content = response.content.replace("\n", "").replace(",", "").replace(" ", "")
        logger.info(f"ROUTING TO ASSISTANT: {response.content}")
        print("ROUTING TO ASSISTANT: ", response.content)
        return response.content
    except Exception as e:
        logger.error(f"Unhandled error in route_input: {str(e)}")
        # Default to a safe assistant type in case of error
        return "FA"


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

multimodal_agentic_rag_graph = multimodal_agentic_rag.compile()

"""
# Create the directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Generate the Mermaid diagram and save it
mermaid_png = multimodal_agentic_rag_graph.get_graph(xray=1).draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)

# Save the image to the assets directory
with open("assets/architecture.png", "wb") as f:
    f.write(mermaid_png)

# Display the image
display(Image(mermaid_png))

response = multimodal_agentic_rag_graph.invoke(
    Chat_Input(
        prompt="Answer the question in the image.",
        image_path="/Users/hashimmuhammadnadeem/Documents/test.png",
        assistant_id="",
        last_ai_message="",
        tranformed_queries=[],
        retrieved_docs=[],
        context=[]
    )
)

print(response)
"""