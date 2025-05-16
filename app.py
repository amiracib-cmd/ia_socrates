import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever

st.set_page_config(page_title="IAssistente Sócrates - Projeto IAgora", layout="centered")
st.title("IAssistente Sócrates - Projeto IAgora")

@st.cache_data(show_spinner=False)
def load_documents(csv_path="bncc.csv"):
    df = pd.read_csv(csv_path)
    return [Document(page_content=row["text"], metadata={"title": row.get("title", "")}) for _, row in df.iterrows()]

@st.cache_resource(show_spinner=False)
def setup_retriever(_docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

docs = load_documents()
retriever = setup_retriever(docs)

pergunta = st.text_input("Digite sua pergunta em português:")

if pergunta:
    with st.spinner("Buscando trechos mais relevantes..."):
        resultados = retriever.get_relevant_documents(pergunta)
        st.subheader("📌 Resposta simulada:")
        resposta = "Baseado nos documentos encontrados, podemos dizer que:\n"
        for doc in resultados:
            resposta += f"- Trecho: _{doc.page_content[:200]}_\n"
        st.markdown(resposta)

        st.subheader("🔎 Fontes:")
        for doc in resultados:
            st.markdown(f"- **{doc.metadata.get('title', '')}**")
