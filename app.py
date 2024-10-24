# --- Required Libraries ---
import os
import warnings
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Updated path
from langchain.vectorstores import FAISS  # Using FAISS instead of Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import tempfile  # For handling temporary file storage

# Suppress warnings
warnings.filterwarnings("ignore")

# Load .env file at the start of the script
load_dotenv()

# Check if the environment variable is loaded correctly
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Debugging: Print the value to check if it's being fetched
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")

if not GEMINI_API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY in the environment variables or .env file.")

# --- Streamlit App Setup ---
def create_streamlit_app():
    st.title("RAG System with FAISS and Gemini")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader with the temp file path
        pdf_loader = PyPDFLoader(temp_file_path)
        pages = pdf_loader.load_and_split()

        # Display the first 1000 characters of the PDF
        st.write("First 5000 characters of the PDF:")
        st.write(pages[0].page_content[:5000])

        # Split the text into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context = "\n\n".join([p.page_content for p in pages])
        texts = text_splitter.split_text(context)

        # Generate embeddings using Google Generative AI
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

        # Create vector store from embeddings using FAISS
        vector_index = FAISS.from_texts(texts, embeddings)

        # Define the QA chain prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know.
        Always say, "Feel free to ask more! 💜" at the end.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create ChatGoogleGenerativeAI instance (LLM)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

        # Set up the QA chain with FAISS and the LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_index.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Streamlit Input for asking questions
        question = st.text_input("Ask a question based on the PDF content:")
        if question:
            result = qa_chain({"query": question})
            st.write("Response:")
            st.write(result['result'])

            # Show source documents
            # st.write("Source documents:")
            # for doc in result['source_documents']:
            #     st.write(doc.page_content[:500])

# Start the Streamlit app
if __name__ == "__main__":
    create_streamlit_app()
