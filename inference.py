from preprocess import *
from preprocess_text import *

embedding_model_global = None


def inference_chroma(chat_model, question):
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context to answer questions accurately and concisely.\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        )
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    llm_response = qa_chain(question)
    print(llm_response['result'])

def inference_faiss(chat_model, question):
    from langchain.chains import LLMChain
    from langchain_together import ChatTogether
    from langchain.prompts import PromptTemplate
    import numpy as np

    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert financial advisor. Use the context to answer questions accurately and concisely.\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer (be specific and avoid hallucinations):"
        )
    )

    qa_chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template
    )

    query_embedding = embedding_model_global.embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=1)

    doc_id = I[0][0]
    document = docstore.search(doc_id)
    context = document.page_content

    answer = qa_chain.run(context=context, question=question, clean_up_tokenization_spaces=False)
    print(answer)

def inference_qdrant(chat_model, question):
    from qdrant_client.http.models import SearchRequest
    from langchain_together import ChatTogether
    import numpy as np

    query_embedding = embedding_model_global.embed_query(question)
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
    Question: {question}
    Answer:
    """

    llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                model=chat_model)

    response = llm.predict(prompt)
    print(response)

def inference_pinecone(chat_model, question):
  import pinecone
  from pinecone import Pinecone
  from langchain_together import ChatTogether
  import numpy as np

  # Initialize Pinecone



  # Step 1: Generate query embedding
  query_embedding = embedding_model_global.embed_query(question)
  query_embedding = np.array(query_embedding)

  # Step 2: Search in Pinecone
    # Replace with your Pinecone index name
  search_results =  pinecone_index.query(
      vector=query_embedding.tolist(),
      top_k=2,  # Retrieve top 2 most relevant results
      include_metadata=True
  )

  # Step 3: Extract context from search results
  # Step 3: Extract context from search results
  # Instead of 'page_content', use 'text' which you used during upsert
  contexts = [result['metadata']['text'] for result in search_results['matches']]

  # Combine contexts for LLM
  context = "\n".join(contexts)

  # Step 4: Prepare prompt for Together.ai
  prompt = f"""
  You are a helpful assistant. Use the following retrieved documents to answer the question:
  Context:
  {context}
  Question: {question}
  Answer:
  """
  #llm=ChatmodelInstantiate(chat_model)
  llm = ChatTogether(api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
                  model=chat_model,  )


  # Step 5: Use Together.ai LLM for generation
  response = llm.predict(prompt)
  print(response)

def inference_weaviate(chat_model, question):
    from langchain_together import ChatTogether
    chat_model = ChatTogether(
        together_api_key="c51c9bcaa6bf7fae3ce684206311564828c13fa2e91553f915fee01d517ccee9",
        model=chat_model,
    )
    from langchain.prompts import ChatPromptTemplate
    template= """You are an expert financial advisor. Use the context to answer questions accurately and concisely:
    Context:
    {context}
    Question: {question}
    Answer(be specific and avoid hallucinations)::
    """
    prompt=ChatPromptTemplate.from_template(template)
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    output_parser=StrOutputParser()
    retriever=vs.as_retriever()
    rag_chain=(
    {"context":retriever,"question":RunnablePassthrough()}
      |prompt
      |chat_model
      |output_parser
    )
    result = rag_chain.invoke(question)
    return result