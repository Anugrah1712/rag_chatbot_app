# -*- coding: utf-8 -*-
"""Freezed code of Final PineconeQdrantFAISSandChroma.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N0xv2v3tc6MFMlQ2sxv1cZkGpI3gxEJ2

### Install dependencies
"""



# Call the function


#!pip install "pinecone[grpc]"

from preprocess import *
from inference import *
#!pip install langchain-weaviate
import streamlit as st
#from webscrape import *
import validators

# Main Streamlit App
st.title("RAG MODEL")

# Use session state to manage the preprocessing_done flag and retriever
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

if "preprocessing_done" not in st.session_state:
    st.session_state.preprocessing_done = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.preprocessing_done:
    st.header("Preprocessing")

    # User inputs for preprocessing
    doc_path = st.file_uploader("Upload your Word Document (.docx)", type=["docx", "pdf"], accept_multiple_files=True)

    st.subheader("Add Links for Web Scraping")
    links = []  # To store all the links
    link_count = st.number_input("Number of Links", min_value=0, max_value=50, value=1, step=1)

    # Generate dynamic text inputs for links
    for i in range(link_count):
        link = st.text_input(f"Enter Link {i+1}", key=f"link_{i}")
        if link:
            links.append(link)
            if validators.url(link):  # Validate link format
                links.append(link)
            else:
                st.error(f"Invalid URL format for Link {i + 1}")



    embedding_models = [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "distilbert-base-nli-stsb-mean-tokens",
            "all-distilroberta-v1",
            "paraphrase-distilroberta-base-v1",
            "stsb-roberta-base",
        ]
    st.markdown("### Choose an Embedding Model:")
    col1, col2 = st.columns(2)
    selected_embedding_model = st.session_state.get("selected_embedding_model", None)

    # Create buttons and highlight selected one
    with col1:
        for model in embedding_models[:len(embedding_models)//2]:
            if st.button(model, key=model):
                st.session_state.selected_embedding_model = model

    with col2:
        for model in embedding_models[len(embedding_models)//2:]:
            if st.button(model, key=model):
                st.session_state.selected_embedding_model = model

    # Display the selected model
    if selected_embedding_model:
        st.markdown(f"*Selected Model:* **{selected_embedding_model}**")

    chunk_size = st.number_input("**Enter chunk size:**", min_value=1, value=2000, step=1)
    chunk_overlap = st.number_input("**Enter chunk overlap:**", min_value=1, value=500, step=1)

    if st.button("Next"):
        if doc_path:
            with st.spinner("Processing... This may take a while."):
                try:
                    # Call the preprocess_vectordbs function directly
                    index, docstore, index_to_docstore_id, vector_store, retriever, pinecone_index,embedding_model_global ,vs= preprocess_vectordbs(
                        doc_path, links,selected_embedding_model, chunk_size, chunk_overlap)

                    st.session_state.preprocessing_done = True  # Persist the flag
                    st.session_state.retriever = retriever
                    st.session_state.index = index
                    st.session_state.docstore = docstore
                    st.session_state.embedding_model_global=embedding_model_global

                    st.session_state.pinecone_index=pinecone_index
                    st.session_state.vs=vs
                    #st.session_state.client=client

                    # Store retriever
                    st.success("Preprocessing Vector DBs Complete! Press Next to Proceed.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide all inputs before running preprocessing.")
else:
    st.header("Inferencing")

   # User inputs for inference
    vectordb_names = ["Pinecone", "Chroma", "FAISS", "Qdrant", "Weaviate"]
    st.markdown("### Choose a Vector Database:")

    # Create 2 columns for displaying the vector databases
    cols = st.columns(5)

    # Loop through the vector databases and display them in the columns
    for i, db_name in enumerate(vectordb_names):
        with cols[i % 5]:  # Distribute databases between the two columns
            if st.button(db_name, key=db_name):
                st.session_state.selected_vectordb = db_name

    # Display the selected vector database
    selected_vectordb = st.session_state.get("selected_vectordb", None)
    if selected_vectordb:
        st.markdown(f"*Selected Vector Database:* **{selected_vectordb}**")

    chat_models = [
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "scb10x/scb10x-llama3-typhoon-v1-5-8b-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ]

    st.markdown("### Choose a Chat Model:")

    # Create 2 columns for displaying the models
    cols = st.columns(1)

    # Loop through the models and display them in the columns
    for i, model in enumerate(chat_models):
        if st.button(model, key=model):
            st.session_state.selected_chat_model = model

    # Display the selected chat model
    selected_chat_model = st.session_state.get("selected_chat_model", None)
    if selected_chat_model:
        st.markdown(f"*Selected Chat Model:* **{selected_chat_model}**")
    #st.markdown("### Enter your Question:")
    #question = st.text_input("")
###GO to next page here

    if st.button("Run Chatbot Inference"):
        st.session_state.button_clicked = True

    if st.button("Reset History"):
        st.session_state.button_clicked = False  # Hide chatbot UI
        st.session_state.messages = []  # Clear chat history

    if st.session_state.button_clicked:
            st.title("Chatbot")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("What is up?"):
                with st.chat_message("user"):
                    st.markdown(prompt)

                st.session_state.messages.append({"role": "user", "content": prompt})

                response = inference(selected_vectordb, selected_chat_model, prompt, st.session_state.retriever,
                                       st.session_state.embedding_model_global, st.session_state.index,
                                       st.session_state.docstore,
                                       st.session_state.pinecone_index,
                                       st.session_state.vs,st.session_state.messages)

                #st.session_state.client

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

