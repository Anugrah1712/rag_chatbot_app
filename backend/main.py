from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from preprocess import preprocess_vectordbs
from webscrape import scrape_data  # Import web scraping function
from inference import inference

app = FastAPI()

# Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Store chatbot configurations
configurations = {
    "embedding_model": None,
    "vector_database": None,
    "chat_model": None,
    "chunk_size": None,
    "chunk_overlap": None,
    "documents": [],
    "web_links": [],
    "retriever": None,
    "index": None,
    "docstore": None,
    "pinecone_index": None,
    "vs": None,
    "messages": []
}

# Define Pydantic model for configurations
class ConfigRequest(BaseModel):
    embedding_model: Optional[str]
    chat_model: Optional[str]
    vector_database: Optional[str]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    web_links: Optional[List[str]]

@app.post("/api/setup_chatbot")
async def setup_chatbot(config: ConfigRequest, background_tasks: BackgroundTasks):
    """Receive configurations from Developer Console and trigger preprocessing."""
    configurations.update(config.dict(exclude_unset=True))

    if configurations["documents"] or configurations["web_links"]:
        background_tasks.add_task(preprocess_data)

    return {"message": "Chatbot setup successful", "configurations": configurations}

@app.post("/api/set_config")
async def set_config(
    embedding_model: str = Form(...),
    chat_model: str = Form(...),
    vector_database: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
    web_links: Optional[List[str]] = Form(None),
    documents: List[UploadFile] = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Handle configuration updates from frontend and process uploaded files."""
    try:
        configurations["embedding_model"] = embedding_model
        configurations["chat_model"] = chat_model
        configurations["vector_database"] = vector_database
        configurations["chunk_size"] = chunk_size
        configurations["chunk_overlap"] = chunk_overlap
        configurations["web_links"] = web_links if web_links else []

        # Store actual file content instead of just filenames
        if documents:
            configurations["documents"] = [
                {"filename": doc.filename, "content": await doc.read()} for doc in documents
            ]

        logging.info("Configurations updated: %s", configurations)

        if configurations["documents"] or configurations["web_links"]:
            background_tasks.add_task(preprocess_data)

        return {"message": "Configuration updated successfully", "configurations": configurations}

    except Exception as e:
        logging.error(f"Error in set_config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_config")
async def get_config():
    """Fetch the current chatbot configurations."""
    return configurations

def preprocess_data():
    """Run preprocessing with the latest configurations."""
    try:
        logging.info("Starting Preprocessing...")

        # Extract document contents
        document_contents = [doc["content"].decode("utf-8") for doc in configurations["documents"]]

        # Scrape web links
        scraped_data = []
        if configurations["web_links"]:
            logging.info("Scraping web links...")
            scraped_data = scrape_web_links(configurations["web_links"])
            logging.info("Web scraping completed.")

        # Combine document contents and scraped data
        all_texts = document_contents + scraped_data

        (
            index, docstore, _, vector_store, retriever,
            pinecone_index, embedding_model_global, vs
        ) = preprocess_vectordbs(
            all_texts,  # Pass combined text data
            configurations["embedding_model"],
            configurations["chunk_size"],
            configurations["chunk_overlap"]
        )

        # Store processed components
        configurations["retriever"] = retriever
        configurations["index"] = index
        configurations["docstore"] = docstore
        configurations["pinecone_index"] = pinecone_index
        configurations["vs"] = vs

        logging.info("Preprocessing completed successfully!")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")

class ChatQuery(BaseModel):
    user_input: str

@app.post("/api/chat")
async def chat_with_bot(query: ChatQuery):
    """Process user queries using the selected models."""
    if not configurations["chat_model"] or not configurations["retriever"]:
        raise HTTPException(status_code=400, detail="Configuration is incomplete!")

    response = inference(
        configurations["vector_database"],
        configurations["chat_model"],
        query.user_input,
        configurations["retriever"],
        configurations["embedding_model"],
        configurations["index"],
        configurations["docstore"],
        configurations["pinecone_index"],
        configurations["vs"],
        configurations["messages"]
    )

    configurations["messages"].append({"role": "user", "content": query.user_input})
    configurations["messages"].append({"role": "assistant", "content": response})

    return {"response": response}
