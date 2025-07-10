import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Cargar variables de entorno (.env)
load_dotenv()

st.title("RAG con FAISS y Google Gemini Embeddings")

uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

def load_pdf(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_obj.getbuffer())
        tmp.flush()
        loader = PDFPlumberLoader(tmp.name)
        documents = loader.load()
    return documents

if uploaded_file:
    documents = load_pdf(uploaded_file)
    st.write(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs_chunks = splitter.split_documents(documents)
    st.write(f"Split into {len(docs_chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Crear vector store FAISS desde docs y embeddings
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    st.write("Vector store FAISS creado con los documentos.")

    query = "Summary of instructions"
    results = vector_store.similarity_search(query, k=3)

    st.subheader("Top matching chunks:")
    for i, doc in enumerate(results):
        st.write(f"--- Result {i+1} ---")
        st.write(doc.page_content[:300])
