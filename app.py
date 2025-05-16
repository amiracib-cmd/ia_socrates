import streamlit as st
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="IAssistente Sócrates - Projeto IAgora", layout="centered")
st.title("IAssistente Sócrates - Projeto IAgora")

@st.cache_data(show_spinner=False)
def load_documents(csv_path="bncc.csv"):
    df = pd.read_csv(csv_path)
    docs = [Document(page_content=row["text"], metadata={"title": row.get("title", "")}) for _, row in df.iterrows()]
    return docs

@st.cache_resource(show_spinner=False)
def setup_rag(_docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

docs = load_documents()
qa = setup_rag(docs)

pergunta = st.text_input("Digite sua pergunta em português:")

if pergunta:
    with st.spinner("Consultando base e gerando resposta..."):
        resultado = qa(pergunta)
        st.subheader("Resposta:")
        st.write(resultado["result"])

        st.subheader("Fontes:")
        for doc in resultado["source_documents"]:
            st.markdown(f"- **{doc.metadata.get('title', '')}**: {doc.page_content[:200]}...")


