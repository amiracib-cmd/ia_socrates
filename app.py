import streamlit as st
import pandas as pd
import requests
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="IAssistente Sócrates - Projeto IAgora", layout="centered")
st.title("IAssistente Sócrates - Projeto IAgora")

HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
HF_MODEL_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

@st.cache_data(show_spinner=False)
def load_documents(csv_path="bncc.csv"):
    df = pd.read_csv(csv_path)
    return [Document(page_content=row["text"], metadata={"title": row.get("title", "")}) for _, row in df.iterrows()]

@st.cache_resource(show_spinner=False)
def setup_retriever():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = setup_retriever()

def gerar_prompt(contexto, pergunta):
    return f"""
Responda à pergunta com base no contexto abaixo.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""

def consultar_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.3,
            "max_new_tokens": 512
        }
    }
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    resultado = response.json()
    if isinstance(resultado, list) and "generated_text" in resultado[0]:
        return resultado[0]["generated_text"].split("Resposta:")[-1].strip()
    else:
        return "Erro na resposta do modelo."

pergunta = st.text_input("Digite sua pergunta em português:")

if pergunta:
    with st.spinner("Consultando base de dados e modelo..."):
        documentos = retriever.get_relevant_documents(pergunta)
        contexto = "\n".join([doc.page_content for doc in documentos])
        prompt = gerar_prompt(contexto, pergunta)
        resposta = consultar_llm(prompt)

        st.subheader("📌 Resposta:")
        st.write(resposta)
