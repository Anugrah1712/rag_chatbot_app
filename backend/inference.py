#Inference.py 

from playwright.sync_api import sync_playwright
from langchain_core.prompts import ChatPromptTemplate


def inference_chroma(chat_model, question, retriever, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely.\n\n"
            "Context: {context}\n\n"
            "{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    llm_response = qa_chain(question_with_history)

    print(llm_response['result'])
    return llm_response['result']


def inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history):
    from langchain.chains import LLMChain
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    import numpy as np

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=(
            "You are an expert financial advisor. Use the context and chat history to answer questions accurately and concisely.\n"
            "Chat History:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        ),
    )

    qa_chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template,
    )

    query_embedding = embedding_model_global.embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=1)

    doc_id = I[0][0]
    document = docstore.search(doc_id)
    context = document.page_content

    answer = qa_chain.run(
        history=history_context, context=context, question=question, clean_up_tokenization_spaces=False
    )
    print(answer)

    return answer


def inference_qdrant(chat_model, question, embedding_model_global, client, chat_history):
    from qdrant_client.http.models import SearchRequest
    from langchain_together import ChatTogether
    import numpy as np

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    query_embedding = embedding_model_global.embed_query(question_with_history)
    query_embedding = np.array(query_embedding)

    search_results = client.search(
        collection_name="text_vectors",
        query_vector=query_embedding,
        limit=2
    )

    contexts = [result.payload['page_content'] for result in search_results]
    context = "\n".join(contexts)

    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents to answer the question:

    Context:
    {context}

    {question_with_history}

    Answer:
    """

    llm = ChatTogether(
        api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model
    )

    response = llm.predict(prompt)
    print(response)
    return response


def inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history):
    import numpy as np
    from langchain_together import ChatTogether

    query_embedding = embedding_model_global.embed_query(question)
    query_embedding = np.array(query_embedding)

    search_results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=2,
        include_metadata=True
    )

    contexts = [result['metadata']['text'] for result in search_results['matches']]
    context = "\n".join(contexts)

    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )

    prompt = f"""
    You are a helpful assistant. Use the following retrieved documents and chat history to answer the question:
    Chat History:
    {formatted_history}

    Context:
    {context}

    Question: {question}
    Answer:
    """

    llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                       model=chat_model)

    response = llm.predict(prompt)
    print(response)
    return response


def inference_weaviate(chat_model, question, vs, chat_history):
    from langchain_together import ChatTogether
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    history_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    question_with_history = f"Chat History:\n{history_context}\n\nNew Question:\n{question}"

    template = """
    You are an expert financial advisor. Use the context and the appended chat history in the question to answer accurately and concisely:

    Context:
    {context}

    {question}

    Answer (be specific and avoid hallucinations):
    """
    prompt = ChatPromptTemplate.from_template(template)

    output_parser = StrOutputParser()

    retriever = vs.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | output_parser
    )

    result = rag_chain.invoke(question_with_history)

    return result


def inference(vectordb_name, chat_model, question, retriever, embedding_model_global, index, docstore, pinecone_index, vs, chat_history):
    if vectordb_name == "Chroma":
        return inference_chroma(chat_model, question, retriever, chat_history)
    elif vectordb_name == "FAISS":
        return inference_faiss(chat_model, question, embedding_model_global, index, docstore, chat_history)
    elif vectordb_name == "Qdrant":
        return inference_qdrant(chat_model, question, embedding_model_global, vs, chat_history)
    elif vectordb_name == "Pinecone":
        return inference_pinecone(chat_model, question, embedding_model_global, pinecone_index, chat_history)
    elif vectordb_name == "Weaviate":
        return inference_weaviate(chat_model, question, vs, chat_history)
    else:
        print("Invalid Choice")
