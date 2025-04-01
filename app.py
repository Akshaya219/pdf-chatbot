import os
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text and images from PDF with better metadata
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    image_metadata = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        
        # Extract text blocks to associate with images
        blocks = page.get_text("blocks")
        text_blocks = [b[4] for b in blocks if b[6] == 0]  # Text blocks only
        
        images = page.get_images(full=True)
        images_per_page[page_num] = []

        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width, height = base_image["width"], base_image["height"]
            image_name = base_image.get("name", "").lower()
            
            # Enhanced filtering to remove unwanted images
            if (
                width < 100 or 
                height < 50 or
                (width * height) < 10000 or  # Area-based filtering
                "check your knowledge" in image_name or
                "header" in image_name or
                "footer" in image_name or
                "logo" in image_name
            ):
                continue  # Skip unwanted images

            # Get image rectangle on page
            img_rect = None
            for img_info in page.get_images(full=True):
                if img_info[0] == xref:
                    for obj in page.get_image_info():
                        if obj["xref"] == xref:
                            img_rect = obj["bbox"]
                            break
                    break
            
            # Find closest text to the image for better context
            nearby_text = ""
            if img_rect:
                # Get text within 100 points of the image boundaries
                expanded_rect = (
                    img_rect[0] - 100, 
                    img_rect[1] - 100,
                    img_rect[2] + 100, 
                    img_rect[3] + 100
                )
                nearby_text = page.get_text("text", clip=expanded_rect)
            
            # If no nearby text, use the whole page text
            if not nearby_text and text_blocks:
                nearby_text = text

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                img_path = tmp_img.name
                images_per_page[page_num].append(img_path)
                
                # Store image metadata for relevance matching
                image_metadata[img_path] = {
                    'page': page_num,
                    'nearby_text': nearby_text,
                    'width': width,
                    'height': height,
                    'area': width * height
                }
    
    doc.close()
    return text_per_page, images_per_page, image_metadata

# Function to index PDF text with page metadata
def index_pdf_text(text_per_page):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={'page': page_num})
            documents.append(doc)
    return FAISS.from_documents(documents, embedding_function)

# Function to query Gemini API with concise prompt
def query_gemini(prompt, context):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Valid model as of March 2025
        response = model.generate_content(
            f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise answer suitable for exam preparation."
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

# Function to compute semantic similarity between query and text
def compute_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    
    # Generate embeddings
    emb1 = embedding_model.encode([text1])[0]
    emb2 = embedding_model.encode([text2])[0]
    
    # Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# Improved function to search PDF and answer with relevant images
def search_pdf_and_answer(query, vector_store, images_per_page, image_metadata):
    # Get relevant document chunks
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(query, context)
    
    # Identify pages containing relevant text
    page_nums = {doc.metadata['page'] for doc in docs}
    
    # Get candidate images from relevant pages
    candidate_images = [img for page_num in page_nums for img in images_per_page.get(page_num, [])]
    
    # No candidate images found
    if not candidate_images:
        return answer, [], "No relevant images found for this query."
    
    # Score images based on semantic similarity between query and nearby text
    image_scores = []
    for img_path in candidate_images:
        metadata = image_metadata.get(img_path, {})
        nearby_text = metadata.get('nearby_text', '')
        
        # Compute similarity between query and text near image
        similarity = compute_similarity(query, nearby_text)
        
        # Add bonus for larger images (often more important diagrams/figures)
        area_factor = min(1.0, metadata.get('area', 0) / 100000) * 0.2
        
        final_score = similarity + area_factor
        image_scores.append((img_path, final_score))
    
    # Sort by relevance score and filter by threshold
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Only keep images with good relevance (threshold can be adjusted)
    threshold = 0.25
    relevant_images = [img for img, score in image_scores if score > threshold]
    
    # Limit to top 3 most relevant images to avoid clutter
    relevant_images = relevant_images[:3]
    
    message = ""
    if not relevant_images:
        message = "No relevant images found for this query."
    
    return answer, relevant_images, message

# Streamlit UI
st.title("📄 PDF Chatbot with Gemini API 🤖")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vector_store = None
    st.session_state.images_per_page = None
    st.session_state.image_metadata = None

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file and not st.session_state.pdf_processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    with st.spinner("Processing PDF... Please wait..."):
        text_per_page, images_per_page, image_metadata = extract_text_and_images_from_pdf(temp_path)
        vector_store = index_pdf_text(text_per_page)
        
        st.session_state.vector_store = vector_store
        st.session_state.images_per_page = images_per_page
        st.session_state.image_metadata = image_metadata
        st.session_state.pdf_processed = True
    st.success("PDF successfully indexed! ✅")

query = st.text_input("Ask a question from the PDF:")

if query and st.session_state.pdf_processed:
    with st.spinner("Generating response..."):
        answer, relevant_images, message = search_pdf_and_answer(
            query, 
            st.session_state.vector_store, 
            st.session_state.images_per_page,
            st.session_state.image_metadata
        )
    
    st.write("### 🤖 Answer:")
    st.write(answer)
    
    if relevant_images:
        st.write("#### Relevant Images from PDF:")
        for img_path in relevant_images:
            st.image(img_path, use_column_width=True)
    elif message:
        st.info(message)
