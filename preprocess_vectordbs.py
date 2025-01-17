from preprocess import *
from preprocess_text import *
embedding_model_global = None

def preprocess_vectordbs(word_file_path, embedding_model_name, size):
    global embedding_model_global  # Declare embedding_model_global as global within the function

    text = preprocess_text(word_file_path, size)

    persist_directory = 'db'
    # Assign the model directly, not the model name
    embedding_model_global = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Process Chroma
    vectordb, retriever =preprocess_chroma(text, embedding_model_name, persist_directory) #embedding_model_name changed to embedding_model_global

    # Process FAISS
    index, docstore, index_to_docstore_id, vector_store = preprocess_faiss(text, embedding_model_name) #embedding_model_name changed to embedding_model_global

    # Process Qdrant
    embeddings = vector_store.index.reconstruct_n(0, len(text))
    client_url = "https://186e02e2-6d10-4b48-baf1-273a91f6c628.us-east4-0.gcp.cloud.qdrant.io:6333"
    client_api_key = "khkhQd22_WZRUBXQg_kL_I08CH3L5HmuHGrbETbVaZlyzCQfyjG0_w"
    collection_name = "text_vectors"
    client = preprocess_qdrant(text, embeddings, client_url, client_api_key, collection_name)
    preprocess_pinecone(text,embedding_model_name)

    #pinecone
    pinecone_index = preprocess_pinecone(text,embedding_model_name)

    #process weaviate
    vs = preprocess_weaviate(text, embedding_model_name)

    return index, docstore, index_to_docstore_id, vector_store, retriever, client,pinecone_index,vs
