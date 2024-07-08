import requests
import streamlit as st

def get_llm_response(input_text):
    response = requests.post("http://localhost:8001/essay/invoke",
                             json={'input':{'topic': input_text}})


    return response.json()['output']


def get_ollama_response(input_text):
    response = requests.post("http://localhost:8001/poem/invoke",
                             json={'input':{'topic': input_text}})
    return response.json()['output']


st.title("LANGCHAIN demo with LLAMA3 and PHI3 API")
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_llm_response(input_text))


if input_text1:
    st.write(get_ollama_response(input_text1))







