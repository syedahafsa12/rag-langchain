# 📖 RAG-Powered Chatbot with FAISS & Gemini AI
![image](https://github.com/user-attachments/assets/55a70bc5-211f-46ae-ba01-04428e7feb8f)

## 🚀 Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using FAISS (Facebook AI Similarity Search) for vector storage and Google Gemini AI for LLM responses. It allows users to upload PDFs, extract content, generate embeddings, and ask AI-powered questions based on the document.

✅ Tech Stack:

LangChain (for orchestration)
FAISS (for vector storage)
Google Gemini AI (for embeddings & responses)
Streamlit (for the frontend UI)
PyPDFLoader (for extracting PDF text)


🎯 Features
✅ Upload a PDF document
✅ AI extracts & indexes the content
✅ Ask questions based on the PDF
✅ AI responds using RAG (Retrieval-Augmented Generation)
✅ Uses FAISS for fast vector similarity search

🛠 How It Works
1️⃣ User uploads a PDF
2️⃣ Text is extracted from the PDF using PyPDFLoader
3️⃣ Text is split into chunks using RecursiveCharacterTextSplitter
4️⃣ Embeddings are generated using Google Gemini AI
5️⃣ FAISS stores the embeddings as a vector database
6️⃣ User asks a question, and the chatbot retrieves relevant context using FAISS
7️⃣ Google Gemini AI generates a response based on the retrieved context
8️⃣ AI responds with helpful answers & references

🖥 Live Demo
🚀 Try it here: hafsa-rag-project.streamlit.app
