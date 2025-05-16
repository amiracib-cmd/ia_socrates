import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os

st.set_page_config(page_title="IAssistente Sócrates - Projeto IAgora", layout="centered")
st.title("IAssistente Sócrates - Projeto IAgora")

@st.cache_data(show_spinner=False)
def load_documents(csv_path="bncc.csv"):
    df = pd.read_csv(csv_path)
    docs = [
        Document(page_content=row["text"], metadata={"title": row["title"]})
        for _, row in df.iterrows()
    ]
    return docs

@st.cache_resource(show_spinner=False)
def setup_rag(_docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(split_docs, embeddings)

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

    prompt_template = PromptTemplate.from_template("""
Você é um assistente inteligente. Responda com base apenas nos trechos abaixo. Use o português claro e objetivo.

Trechos do contexto:
{context}

Pergunta:
{question}

Resposta:
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

try:
    docs = load_documents("bncc.csv")
    qa_chain = setup_rag(docs)
    ready = True
except Exception as e:
    st.error(f"Erro ao inicializar o sistema: {e}")
    ready = False

if ready:
    query = st.text_input("Digite sua pergunta sobre o conteúdo:")
    if query:
        with st.spinner("Consultando a base com LLaMA 3..."):
            answer = qa_chain.invoke(query)
            st.success(answer)

