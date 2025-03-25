# Lab-Building-RAG-Systems

![COVER_-_EMAIL](https://github.com/user-attachments/assets/fa63da36-0a2d-4e4b-a11e-ceb163c31ae1)

## Submission

1. Push your completed Langchain script to your GitHub repository.
2. Create a Pull Request with a description of your implementations.

---

# Introduction

This lab develops an interactive web application that allows users to upload PDFs, extract their content, index it with embeddings, and perform natural language queries. Langchain and Streamlit tools are integrated to create a chat interface for querying information from PDFs.

---

# Requirements

Before starting, configure the environment and install the necessary libraries:

```bash
pip install streamlit langchain_community langchain_text_splitters pdfplumber langchain_ollama langchain_core
```

---

## Iteration 1: PDF Upload and Extraction

Use Streamlit to create an input for uploading PDF files and `PDFPlumberLoader` to load and extract its content. Additionally, use `tempfile` to temporarily save the file:

```python
import streamlit as st
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader

st.title("Upload PDF")
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    documents = load_pdf(uploaded_file)
  

def load_pdf(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_obj.getbuffer())
        tmp.flush()  
        loader = PDFPlumberLoader(tmp.name)
        documents = loader.load()
    return documents
```

---

## Iteration 2: Text Segmentation

Once the PDF content is accessible, segment it into smaller, manageable text chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

## Iteration 3: Content Indexing

Now, add embeddings to the `vector_store` and store the segmented document:

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="bge-m3:latest")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents)
```

---

## Iteration 4: Vector Retrieval

After storing the information, retrieve stored fragments and verify that the information is correctly returned:

```python
vector_store.similarity_search("Summary of instructions")
```

---

## Iteration 5: Response Generation

Using a locally installed Ollama model such as `deepseek-r1:8b` or a Hugging Face model via their API inference, generate a response based on the PDF content.

Optional prompt:

```python
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
```

---

## Iteration 6: Full Integration

Integrate all functionalities, allowing users to enter queries about the PDF in an input field, so the model can provide relevant answers on the web interface.



## 🔥 Bonus: Web Interface

Enhance your **Assistant** by creating a **web interface** with **HTML** for real-time chat.



## Additional Resources

- [Langchain Documentation](https://docs.langchain.com/)
- [LangChain Ollama Installation Guide](https://python.langchain.com/docs/integrations/llms/ollama/)

![Happy_coding](https://github.com/user-attachments/assets/bbf0ccdc-0137-461f-b3c6-1b530016b390)

