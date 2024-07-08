import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import time


from dotenv import load_dotenv
load_dotenv()


## Load Groq api key

groq_Api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)
    st.session_state.loader = WebBaseLoader("https://kalypso.com/services/data-science/case-study-making-formula-management-a-piece-of-cake-with-ai")

    st.session_state.docs =  st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("demo using LLAMA3 using GROQ")

llm = ChatGroq(groq_api_key = groq_Api_key, model_name = "llama3-8b-8192" )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the quetions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    
    Questions " {input}
    """

)

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vectors.as_retriever()
retriver_chain = create_retrieval_chain(retriver,document_chain)


prompt = st.text_input("input your prompt here")

if prompt:
    start = time.process_time()
    response = retriver_chain.invoke({"input": prompt})
    print("Response Time :" , time.process_time() - start)

    st.write(response['answer'])

    # with a streamlit expander

    with st.expander("Document Simililarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------------------------------")

