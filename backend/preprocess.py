#Preprocess.py 

import streamlit as st
# import chromadb
# chromadb.api.client.SharedSystemClient.clear_system_cache()

from playwright.sync_api import sync_playwright
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

def preprocess_text(files, links, size, overlap):
    import time
    
    paragraphs = []

    # Step 1: Process each file
    for file in files:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraphs.extend(page_text.split("\n"))
        elif file.filename.endswith(".docx"):
            docx = DocxDocument(file)
            for paragraph in docx.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)

    # Step 2: Use Playwright for web scraping
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for link in links:
            try:
                page.goto(link, timeout=60000)
                time.sleep(3)  # Allow page to load
                body_text = page.text_content("body")
                paragraphs.extend(body_text.split("\n"))

                # Extract FAQs
                try:
                    faq_container = page.query_selector(".faqs.aem-GridColumn.aem-GridColumn--default--12")
                    if faq_container:
                        show_more_button = faq_container.query_selector(".accordion_toggle_show-more")
                        if show_more_button and show_more_button.is_visible():
                            show_more_button.click()
                            time.sleep(1)

                        toggle_buttons = faq_container.query_selector_all(".accordion_toggle, .accordion_row")
                        all_faqs = []
                        
                        for button in toggle_buttons:
                            button.click()
                            time.sleep(1)
                            expanded_content = faq_container.query_selector_all(".accordion_body, .accordionbody_links, .aem-rte-content")
                            
                            for content in expanded_content:
                                text = content.text_content().strip()
                                if text and text not in [faq['answer'] for faq in all_faqs]:
                                    question = button.text_content().strip()
                                    if question:
                                        all_faqs.append({"question": question, "answer": text})

                        print("Extracted FAQ Questions and Answers:")
                        for i, faq in enumerate(all_faqs, start=1):
                            print(f"Q: {faq['question']}\n   A: {faq['answer']}\n")

                except Exception as faq_error:
                    print(f"FAQ extraction failed for {link}: {faq_error}")
            except Exception as link_error:
                print(f"Failed to process link {link}: {link_error}")

        browser.close()

    # Step 3: Remove empty paragraphs
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    # Step 4: Convert paragraphs to Document objects
    docs = [LangchainDocument(page_content=para) for para in paragraphs]

    # Step 5: Use RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    text_chunks = text_splitter.split_documents(docs)

    return text_chunks



def preprocess_chroma(text, embedding_model_name, persist_directory):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.vectorstores import Chroma

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma.from_documents(documents=text, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()

    return vectordb, retriever

def preprocess_faiss(text, embedding_model_name):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    import numpy as np
    import faiss
    from langchain.docstore.in_memory import InMemoryDocstore
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in text]
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    docstore = InMemoryDocstore({i: Document(page_content=texts[i]) for i in range(len(texts))})
    index_to_docstore_id = {i: i for i in range(len(texts))}

    vector_store = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model.embed_query
    )

    return index, docstore, index_to_docstore_id, vector_store

def preprocess_qdrant(text, embeddings, client_url, client_api_key, collection_name, batch_size=250):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=client_url, api_key=client_api_key)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE)
    )

    qdrant_index = list(range(1, len(text) + 1))
    for i in range(0, len(text), batch_size):
        low_idx = min(i + batch_size, len(text))
        batch_of_ids = qdrant_index[i: low_idx]
        batch_of_embs = embeddings[i: low_idx]
        batch_of_payloads = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in text[i: low_idx]]

        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=batch_of_ids,
                vectors=batch_of_embs.tolist(),
                payloads=batch_of_payloads
            )
        )

    return client

def preprocess_pinecone(text,embedding_model_name):
    import numpy as np
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embedding_model= SentenceTransformerEmbeddings(model_name=embedding_model_name)
    # Extract the 'page_content' (text) from each Document object
    texts = [doc.page_content for doc in text]
    embeddings = embedding_model.embed_documents(texts)  # Pass the list of texts
    embeddings = np.array(embeddings)
    embeddings = embeddings.tolist()


    import pinecone
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    import uuid

    # ... (your existing code) ...

    index_name = "test5"

    pinecone = Pinecone(
        api_key="pcsk_42Yw14_EaKdaMLiAJfWub3s2sEJYPW3jyXXjdCYkH8Mh8rD8wWJ3pS6oCCC9PGqBNuDTuf",
        environment="us-east-1"
    )
    # Check if the index exists
    indexes = pinecone.list_indexes().names()

    if index_name in indexes:
        pinecone.delete_index(index_name)

    pinecone.create_index(
      name=index_name,
      dimension=len(embeddings[0]),
      metric="cosine",
      spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
      ),
      deletion_protection="disabled"
    )

    pinecone_index = pinecone.Index(index_name)

    upsert_data = []
    for i in range(len(texts)):
      upsert_data.append((str(uuid.uuid4()), embeddings[i], {"text": texts[i]}))

    # Upsert data in batches (adjust batch_size as needed)
    batch_size = 100  # Example batch size
    for i in range(0, len(upsert_data), batch_size):
        batch = upsert_data[i : i + batch_size]
        pinecone_index.upsert(vectors=batch)
    return pinecone_index
    # ... (rest of your upsert code) ...

#!pip install weaviate-client

import numpy as np
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
import weaviate
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wvcc
from weaviate.collections import Collection

def preprocess_weaviate(text, embedding_model_name):
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    import numpy as np
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    import os
    os.environ["WEAVIATE_URL"] = "https://pdarzyhgqows9ocn5oava.c0.asia-southeast1.gcp.weaviate.cloud"
    os.environ["WEAVIATE_API_KEY"] = "u5qU5QbMDcvKw8pPewXXAcmHNvRYNstmOxES"

    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
    import weaviate
    from weaviate.auth import AuthApiKey

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_api_key),
    )
    from langchain_weaviate.vectorstores import WeaviateVectorStore # Import WeaviateVectorStore

    vs = WeaviateVectorStore.from_documents(
        documents=text,
        embedding=embedding_model,
        client=client
    )

    return vs


# ipython-input-12-291a5c8eb7ff
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Declare embedding_model_global as a global variable
embedding_model_global = None

def preprocess_vectordbs(files,links, embedding_model_name, size, overlap):
    import sys
    import sqlite3

    sys.modules["sqlite3"] = sqlite3


    global embedding_model_global  # Declare embedding_model_global as global within the function

    text = preprocess_text(files,links, size,overlap)
    st.success("Preprocessing Text Complete!")
    persist_directory = 'db'
    # Assign the model directly, not the model name
    embedding_model_global = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Process Chroma
    vectordb, retriever = preprocess_chroma(text, embedding_model_name, persist_directory) #embedding_model_name changed to embedding_model_global
    st.success("Preprocessing Chroma Complete!")
    # Process FAISS
    index, docstore, index_to_docstore_id, vector_store = preprocess_faiss(text, embedding_model_name) #embedding_model_name changed to embedding_model_global
    st.success("Preprocessing Faiss Complete!")
    # Process Qdrant UNMASK LATER
    # pinecone
    vs = preprocess_weaviate(text, embedding_model_name)
    st.success("Preprocessing Weaviate Complete!")

    pinecone_index = preprocess_pinecone(text, embedding_model_name)
    st.success("Preprocessing Pinecone Complete!")
    # process weaviate

    # return vs for weaviate
    #embeddings = vector_store.index.reconstruct_n(0, len(text))
    #client_url = "https://186e02e2-6d10-4b48-baf1-273a91f6c628.us-east4-0.gcp.cloud.qdrant.io"
    #client_api_key = "Wc7kgaf6hXuYIHppaAT87CUyVy5pwigwGaI3oufb3r3Xbcwdo9c_jw"
    #collection_name = "text_vectors"
    #client = preprocess_qdrant(text, embeddings, client_url, client_api_key, collection_name)
    # st.success("Preprocessing Qdrant Complete!")



    return index, docstore, index_to_docstore_id, vector_store, retriever, pinecone_index,embedding_model_global,vs



