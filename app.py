import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="📚 RAG Q&A App", layout="centered")
st.title("📖 Simple RAG-based Q&A Chatbot")
st.write("Upload a text file and ask questions about its content.")

# -----------------------------
# Step 1: Upload file
# -----------------------------
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    # Save the uploaded file locally
    with open("uploaded.txt", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded successfully!")

    # -----------------------------
    # Step 2: Load and split text
    # -----------------------------
    loader = TextLoader("uploaded.txt", encoding="utf-8")
    text = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    text_splitter = splitter.split_documents(text)

    # -----------------------------
    # Step 3: Embedding + FAISS
    # -----------------------------
    embediing = OllamaEmbeddings(model="nomic-embed-text:latest")
    db = FAISS.from_documents(text_splitter, embedding=embediing)

    retriever = db.as_retriever()

    # -----------------------------
    # Step 4: LLM + RetrievalQA
    # -----------------------------
    model = Ollama(model="mistral:latest")
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

    # -----------------------------
    # Step 5: Question Input
    # -----------------------------
    question = st.text_input("Ask a question from the uploaded file:")

    if st.button("Get Answer"):
        if question.strip() != "":
            with st.spinner("Thinking... 🤔"):
                response = qa_chain.invoke(question)
                st.success(response["result"])
        else:
            st.warning("Please enter a question.")
