import os
import time
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from playwright.sync_api import sync_playwright
from langchain_community.embeddings import SentenceTransformerEmbeddings


def preprocess_text(files, links, size, overlap):
    paragraphs = []

    # Step 1: Process each file
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    paragraphs.extend(page_text.split("\n"))
        elif file.name.endswith(".docx"):
            docx = DocxDocument(file)
            for paragraph in docx.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)

    # Step 2: Process each link using Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for link in links:
            try:
                page.goto(link, timeout=10000)  # Load page
                time.sleep(3)  # Allow content to render

                body_text = page.inner_text("body")
                paragraphs.extend(body_text.split("\n"))

                # Extract FAQ sections
                try:
                    faq_selectors = [
                        ".faqs.aem-GridColumn.aem-GridColumn--default--12",
                        ".accordion_toggle",
                        ".accordion_row",
                        ".accordion_body",
                        ".accordionbody_links",
                        ".aem-rte-content"
                    ]

                    for selector in faq_selectors:
                        elements = page.query_selector_all(selector)
                        for element in elements:
                            text = element.inner_text().strip()
                            if text:
                                paragraphs.append(text)

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
    from langchain.vectorstores import Chroma

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vectordb = Chroma.from_documents(documents=text, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist()

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()

    return vectordb, retriever


def preprocess_faiss(text, embedding_model_name):
    import numpy as np
    import faiss
    from langchain.docstore.in_memory import InMemoryDocstore
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in text]
    embeddings = np.array(embedding_model.embed_documents(texts))
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


def preprocess_pinecone(text, embedding_model_name):
    import numpy as np
    import uuid
    from pinecone import Pinecone, ServerlessSpec

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    texts = [doc.page_content for doc in text]
    embeddings = np.array(embedding_model.embed_documents(texts)).tolist()

    index_name = "test5"
    pinecone = Pinecone(
        api_key="your_pinecone_api_key",
        environment="us-east-1"
    )

    indexes = pinecone.list_indexes().names()
    if index_name in indexes:
        pinecone.delete_index(index_name)

    pinecone.create_index(
        name=index_name,
        dimension=len(embeddings[0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled"
    )

    pinecone_index = pinecone.Index(index_name)
    upsert_data = [(str(uuid.uuid4()), embeddings[i], {"text": texts[i]}) for i in range(len(texts))]

    batch_size = 100
    for i in range(0, len(upsert_data), batch_size):
        batch = upsert_data[i: i + batch_size]
        pinecone_index.upsert(vectors=batch)

    return pinecone_index


def preprocess_weaviate(text, embedding_model_name):
    import os
    from weaviate.auth import AuthApiKey
    from weaviate.collections import Collection
    from langchain_weaviate.vectorstores import WeaviateVectorStore

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    weaviate_url = "your_weaviate_url"
    weaviate_api_key = "your_weaviate_api_key"

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_api_key),
    )

    vs = WeaviateVectorStore.from_documents(
        documents=text,
        embedding=embedding_model,
        client=client
    )

    return vs


def preprocess_vectordbs(files, links, embedding_model_name, size, overlap):
    import sqlite3
    import sys

    sys.modules["sqlite3"] = sqlite3

    text = preprocess_text(files, links, size, overlap)
    persist_directory = 'db'

    embedding_model_global = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    vectordb, retriever = preprocess_chroma(text, embedding_model_name, persist_directory)
    index, docstore, index_to_docstore_id, vector_store = preprocess_faiss(text, embedding_model_name)
    vs = preprocess_weaviate(text, embedding_model_name)
    pinecone_index = preprocess_pinecone(text, embedding_model_name)

    return index, docstore, index_to_docstore_id, vector_store, retriever, pinecone_index, embedding_model_global, vs
