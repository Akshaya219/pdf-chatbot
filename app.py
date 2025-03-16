import streamlit as st
import tempfile
import os

# Try importing fitz correctly
try:
    import fitz  # Standard way when PyMuPDF is installed correctly
except ImportError:
    try:
        from PyMuPDF import fitz  # Alternative import path
    except ImportError:
        st.error("Failed to import PyMuPDF. Please check your installation.")
        st.stop()

# Import other libraries
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Configure API with environment variable (no hardcoded key)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Make sure to set the GOOGLE_API_KEY environment variable.")
    st.stop()
genai.configure(api_key=api_key)

# Rest of your code remains similar
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def index_pdf_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(texts, embedding_function)
    return vector_store

def query_gemini(prompt, context):
    try:
        # Use a valid model name (gemini-pro is commonly available)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"Context: {context}\nUser Query: {prompt}")
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

def search_pdf_and_answer(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)
    return answer

st.title("📄 PDF Chatbot with Gemini API 🤖")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    st.info("Processing PDF... Please wait...")
    pdf_text = extract_text_from_pdf(temp_path)
    vector_store = index_pdf_text(pdf_text)
    st.success("PDF successfully indexed! ✅")
    query = st.text_input("Ask a question from the PDF:")
    if query:
        answer = search_pdf_and_answer(query, vector_store)
        st.write("### 🤖 Answer:")
        st.write(answer)
