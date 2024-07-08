import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import time
from langchain_community.llms import Ollama
import pandas as pd
import numpy as np

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader



from dotenv import load_dotenv

load_dotenv()


class JSONLoader(BaseLoader):
    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
    ):
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key

    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""

        docs = []
        # Load JSON file
        with open(self.file_path) as file:
            data = json.load(file)

            # Iterate through 'pages'
            for column in data['columns']:
                colname = column['name']
                coldesc = column['Description']
                metadata = {"description": coldesc}

                docs.append(Document(page_content=colname, metadata=metadata))
            print(docs)
        return docs


# ## Load Groq api key
#
# groq_Api_key = os.environ['GROQ_API_KEY']


if __name__ == '__main__':

    file_path = 'columns_description.json'
    loader = JSONLoader(file_path=file_path)
    data = loader.load()



    if "vector" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)
        st.session_state.loader = loader = JSONLoader(file_path=file_path)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    st.title("demo using LLAMA3 to generate SQL Queries")

    llm = Ollama(model = "llama3")

    prompt = ChatPromptTemplate.from_template("""
    Write MS SQL queries on table "RVC_FML" based only on the provided context.
    Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>

    Question : {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriver = st.session_state.vectors.as_retriever()
    retriver_chain = create_retrieval_chain(retriver, document_chain)

    prompt = st.text_input("input your prompt here")

    if prompt:
        start = time.process_time()
        response = retriver_chain.invoke({"input": prompt})
        print("Response Time :", time.process_time() - start)

        st.write(response['answer'])

        # with a streamlit expander

        with st.expander("Document Simililarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("------------------------------------------------------")

