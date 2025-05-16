import streamlit as st
import pandas as pd
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


st.set_page_config(page_title="IAssistente Sócrates - Projeto IAgora", layout="centered")
st.title("IAssistente Sócrates - Projeto IAgora")

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

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

def gerar_prompt(contexto, pergunta):
    return f"""
Você é um assistente inteligente que responde em português com base em documentos da BNCC. Use o seguinte contexto para responder à pergunta.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:"""

def consultar_grok(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "xai/grok-1:free",
        "messages": [
            {"role": "system", "content": "Você é um assistente útil que responde em português."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

retriever = setup_retriever()
pergunta = st.text_input("Digite sua pergunta em português:")

if pergunta:
    with st.spinner("Consultando base de dados e modelo Grok-1..."):
        documentos = retriever.invoke(pergunta)
        contexto = "\n".join([doc.page_content for doc in documentos])
        prompt = gerar_prompt(contexto, pergunta)
        resposta = consultar_grok(prompt)

        st.subheader("📌 Resposta:")
        st.write(resposta)
