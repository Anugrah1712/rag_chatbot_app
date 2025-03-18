#main.py 

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from typing import List
from preprocess import preprocess_vectordbs
from inference import inference
from webscrape import scrape_web_data
import validators
import uvicorn
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import os 
import pickle 
from pinecone import Pinecone
import os 
from dotenv import load_dotenv
import weaviate

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to match your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# File path for saved session state
PICKLE_FILE_PATH = "session_state.pkl"

# Load previous session if exists
if os.path.exists(PICKLE_FILE_PATH):
    with open(PICKLE_FILE_PATH, "rb") as f:
        session_state = pickle.load(f)
        print("✅ Loaded saved session state!")

    # Reinitialize Pinecone dynamically
    if session_state.get("pinecone_index_name"):
        pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
        session_state["pinecone_index"] = pinecone.Index(session_state["pinecone_index_name"])

    # ✅ Ensure last selected vector DB and chat model are retained
    session_state.setdefault("selected_vectordb", session_state.get("selected_vectordb", "FAISS"))
    session_state.setdefault("selected_chat_model", session_state.get("selected_chat_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"))

        # ✅ Initialize Weaviate Client if Needed
    if session_state.get("vs"):
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        session_state["weaviate_client"] = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.AuthApiKey(weaviate_api_key),
        )

    # ✅ Initialize Qdrant Client if Needed
    if session_state.get("qdrant_client") is None:
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        session_state["qdrant_client"] = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)    

else:
    session_state = {
        "retriever": None,
        "preprocessing_done": False,
        "index": None,
        "docstore": None,
        "embedding_model_global": None,
        "pinecone_index_name": None,
        "pinecone_index": None,
        "vs": None,
        "qdrant_client": None,
        "weaviate_client": None,
        "selected_vectordb": "FAISS",
        "selected_chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": []
    }

@app.post("/preprocess")
async def preprocess(
    doc_files: List[UploadFile] = File(...),
    links: str = Form(...),
    embedding_model: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...)
):
    """ Preprocessing: Handle document uploads and web scraping """
    
    try:
        print("\n🔍 Preprocessing Started...")
        print(f"📂 Received {len(doc_files)} document(s)")
        print(f"🔗 Links received: {links}")
        print(f"📊 Embedding Model: {embedding_model}")
        print(f"🔢 Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

        # Validate links
        links_list = json.loads(links)
        for link in links_list:
            if not validators.url(link):
                raise HTTPException(status_code=400, detail=f"❌ Invalid URL: {link}")

        # Validate uploaded files
        if not doc_files and not links_list:
            raise HTTPException(status_code=400, detail="❌ No documents or links provided for preprocessing!")

        for file in doc_files:
            if file.filename == "":
                raise HTTPException(status_code=400, detail="❌ One of the uploaded files is empty!")

        # Web scraping
        if links_list:
            try:
                print("🌐 Scraping web data...")
                scraped_data = await scrape_web_data(links_list)
                print("✅ Web scraping completed!\n")
            except Exception as e:
                print(f"❌ Web scraping failed: {str(e)}\n")
                raise HTTPException(status_code=500, detail=f"Web scraping failed: {str(e)}")

    
        # Process documents
        try:
            index, docstore, index_to_docstore_id, vector_store, retriever, embedding_model_global, pinecone_index_name , vs ,qdrant_client= await preprocess_vectordbs(
            doc_files, scraped_data , embedding_model, chunk_size, chunk_overlap
            )

            session_state.update({
                "retriever": retriever,
                "preprocessing_done": True,
                "index": index,
                "docstore": docstore,
                "embedding_model_global": embedding_model_global,
                "pinecone_index_name": pinecone_index_name,  # ✅ Now correctly defined
                "vs": vs ,
                "qdrant_client": qdrant_client
            })


           # **Save state to pickle file (excluding non-pickleable objects)**
            session_state_to_save = session_state.copy()
            session_state_to_save.pop("retriever", None)
            session_state_to_save.pop("index", None)
            session_state_to_save.pop("docstore", None)
            session_state_to_save.pop("pinecone_index", None) 
            session_state_to_save.pop("vs", None)
            session_state_to_save.pop("qdrant_client", None)

            with open(PICKLE_FILE_PATH, "wb") as f:
                pickle.dump(session_state_to_save, f)

            print("💾 Session state saved (excluding non-pickleable objects)!")
            return {"message": "Preprocessing completed successfully!"}

        except Exception as e:
            print(f"❌ Error in preprocess_vectordbs: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.post("/select_vectordb")
async def select_vectordb(vectordb: str = Form(...)):
    """ Set selected vector database and persist it """
    session_state["selected_vectordb"] = vectordb

    # ✅ Save state to pickle file (excluding unpicklable objects)
    session_state_to_save = session_state.copy()
    session_state_to_save.pop("retriever", None)
    session_state_to_save.pop("index", None)
    session_state_to_save.pop("docstore", None)
    session_state_to_save.pop("pinecone_index", None)  # ✅ Exclude Pinecone
    session_state_to_save.pop("vs", None)
    session_state_to_save.pop("qdrant_client", None)

    with open(PICKLE_FILE_PATH, "wb") as f:
        pickle.dump(session_state_to_save, f)

    print(f"✅ Selected Vector Database: {vectordb} (Saved to session state)\n")
    return {"message": f"Selected Vector Database: {vectordb}"}



@app.post("/select_chat_model")
async def select_chat_model(chat_model: str = Form(...)):
    """ Set selected chat model and persist it """
    session_state["selected_chat_model"] = chat_model

    # ✅ Save state to pickle file (excluding unpicklable objects)
    session_state_to_save = session_state.copy()
    session_state_to_save.pop("retriever", None)
    session_state_to_save.pop("index", None)
    session_state_to_save.pop("docstore", None)
    session_state_to_save.pop("pinecone_index", None)  # ✅ Exclude Pinecone
    session_state_to_save.pop("vs", None)
    session_state_to_save.pop("qdrant_client", None)  # ✅ Exclude Qdrant client


    with open(PICKLE_FILE_PATH, "wb") as f:
        pickle.dump(session_state_to_save, f)

    print(f"✅ Selected Chat Model: {chat_model} (Saved to session state)\n")
    return {"message": f"Selected Chat Model: {chat_model}"}




@app.post("/chat")
async def chat_with_bot(prompt: str = Form(...)):
    """ Chatbot interaction """
    if not session_state["preprocessing_done"]:
        raise HTTPException(status_code=400, detail="❌ Preprocessing must be completed before inferencing.")

    session_state["selected_vectordb"] = session_state.get("selected_vectordb", "FAISS")
    session_state["selected_chat_model"] = session_state.get("selected_chat_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")



    # Store user message
    session_state["messages"].append({"role": "user", "content": prompt})

    pinecone_index_name = session_state.get("pinecone_index_name", None)
    vs = session_state.get("vs", None)
    qdrant_client = session_state.get("qdrant_client", None)

    # # ✅ Ensure Pinecone index is reloaded if needed
    # if session_state["selected_vectordb"] == "Pinecone" and pinecone_index_name:
    #     pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
    #     session_state["pinecone_index"] = pinecone.Index(pinecone_index_name)
    # Run inference
    try:
        response = inference(
        session_state["selected_vectordb"],
        session_state["selected_chat_model"],
        prompt,
        session_state["embedding_model_global"],
        session_state["messages"],
        pinecone_index_name,
        vs,
        qdrant_client
        )

        # Store assistant response
        session_state["messages"].append({"role": "assistant", "content": response})

        print(f"🤖 Chatbot Response: {response}\n")
        return {"response": response}

    except Exception as e:
        print(f"❌ Error in inference: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.post("/reset")
async def reset_chat():
    """ Reset chatbot history and delete saved state """
    session_state["messages"] = []
    session_state["preprocessing_done"] = False
    session_state["retriever"] = None
    session_state["index"] = None
    session_state["docstore"] = None
    session_state["embedding_model_global"] = None
    # session_state["pinecone_index"] = None
    # session_state["vs"] = None

    # Delete the saved session file
    if os.path.exists(PICKLE_FILE_PATH):
        os.remove(PICKLE_FILE_PATH)
        print("🗑️ Saved session state deleted!")

    return {"message": "Chat history reset and session state cleared!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
