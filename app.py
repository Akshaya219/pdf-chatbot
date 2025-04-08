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
import base64

# Configure API with environment variable
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Set GOOGLE_API_KEY in Streamlit Cloud secrets.")
    st.stop()
genai.configure(api_key=api_key)

# Initialize models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text and images from PDF with better metadata and contextual understanding
def extract_text_and_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = []
    images_per_page = {}
    image_metadata = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_per_page.append((page_num, text))
        
        # Get structured text with more detail
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])
        
        # Get text by regions for better context association
        text_regions = []
        for block in blocks:
            if block.get("type") == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                if block_text:
                    rect = block.get("bbox")
                    text_regions.append((block_text, rect))
        
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
                (width * height) < 12000 or  # Increased area threshold
                "check" in image_name or
                "header" in image_name or
                "footer" in image_name or
                "logo" in image_name or
                "icon" in image_name or
                "bullet" in image_name or
                "button" in image_name
            ):
                continue  # Skip unwanted images
            
            # For better context, use surrounding paragraphs
            # We'll use the page text but also remember which page this came from
            # for later cross-referencing
            surrounding_text = text
            
            # Caption detection: text immediately before or after image
            caption = ""
            # Simple heuristic: look for text with terms like "figure", "diagram", "image"
            caption_indicators = ["figure", "fig", "diagram", "image", "chart", "graph", "table", "illustration"]
            
            # Check for captions in nearby text
            for text_block, _ in text_regions:
                lower_text = text_block.lower()
                if any(indicator in lower_text for indicator in caption_indicators):
                    caption = text_block
                    break

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                img_path = tmp_img.name
                images_per_page[page_num].append(img_path)
                
                # Store image metadata for relevance matching
                image_metadata[img_path] = {
                    'page': page_num,
                    'surrounding_text': surrounding_text,
                    'caption': caption,
                    'width': width,
                    'height': height,
                    'area': width * height,
                    'aspect_ratio': width / height if height > 0 else 0
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
    
    try:
        # Generate embeddings
        emb1 = embedding_model.encode([text1])[0]
        emb2 = embedding_model.encode([text2])[0]
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)  # Ensure it's a Python float
    except Exception as e:
        st.error(f"Error computing similarity: {str(e)}")
        return 0.0

# Function to rank images by two-step relevance
def rank_images_by_relevance(query, candidate_images, image_metadata, context_text):
    if not candidate_images:
        return []
    
    # First pass: compute similarity scores
    image_scores = []
    
    # Also compute similarity between query and overall context
    query_context_similarity = compute_similarity(query, context_text)
    
    for img_path in candidate_images:
        metadata = image_metadata.get(img_path, {})
        
        # Get text associated with this image
        surrounding_text = metadata.get('surrounding_text', '')
        caption = metadata.get('caption', '')
        
        # Compute multiple similarity scores
        text_similarity = compute_similarity(query, surrounding_text)
        caption_similarity = compute_similarity(query, caption) if caption else 0
        
        # Area-based score (normalized to 0-0.2 range)
        area = metadata.get('area', 0)
        area_factor = min(0.2, area / 500000)
        
        # Caption bonus - images with relevant captions are usually more important
        caption_bonus = 0.15 if caption_similarity > 0.4 else 0
        
        # Check if image is likely a diagram/chart based on aspect ratio
        aspect_ratio = metadata.get('aspect_ratio', 0)
        is_likely_diagram = 0.7 < aspect_ratio < 1.5  # Many diagrams are roughly square-ish
        diagram_bonus = 0.1 if is_likely_diagram else 0
        
        # Compute final score
        # Weight caption similarity more highly if present
        final_score = text_similarity * 0.6 + caption_similarity * 0.3 + area_factor + caption_bonus + diagram_bonus
        
        # Additional check: if query-context similarity is high but image similarity is low, 
        # this might be an irrelevant image on a relevant page
        if query_context_similarity > 0.6 and text_similarity < 0.3:
            final_score *= 0.5  # Penalize
        
        image_scores.append((img_path, final_score))
    
    # Sort by score (descending)
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Apply a higher threshold for acceptance
    threshold = 0.35  # Increased from previous 0.25
    relevant_images = [(img, score) for img, score in image_scores if score > threshold]
    
    # Second pass: filter out very similar images (avoid duplicates)
    unique_images = []
    for i, (img1, score1) in enumerate(relevant_images):
        is_unique = True
        for img2, _ in unique_images:
            metadata1 = image_metadata.get(img1, {})
            metadata2 = image_metadata.get(img2, {})
            
            # If images are from same page and have similar size/position, likely duplicates
            if (
                metadata1.get('page') == metadata2.get('page') and
                abs(metadata1.get('area', 0) - metadata2.get('area', 0)) / max(metadata1.get('area', 1), 1) < 0.2
            ):
                is_unique = False
                break
        
        if is_unique:
            unique_images.append((img1, score1))
    
    # Limit to top 2 most relevant images to avoid irrelevant ones
    return [img for img, _ in unique_images[:2]]

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
    
    # Use advanced ranking function
    relevant_images = rank_images_by_relevance(query, candidate_images, image_metadata, context)
    
    message = ""
    if not relevant_images:
        message = "No relevant images found for this query."
    
    return answer, relevant_images, message

# JavaScript for voice recognition
def get_speech_recognition_js():
    return """
    <script>
    const micButton = document.getElementById('mic-button');
    let recognition;
    let isListening = false;
    
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        
        recognition.onstart = function() {
            document.getElementById('mic-status').textContent = '🎙️ Listening...';
            document.getElementById('mic-button').classList.add('listening');
        };
        
        recognition.onresult = function(event) {
            const transcript = Array.from(event.results)
                .map(result => result[0])
                .map(result => result.transcript)
                .join('');
                
            if (event.results[0].isFinal) {
                document.getElementById('query-input').value = transcript;
                document.getElementById('query-status').textContent = 'Query: ' + transcript;
                stopListening();
                // Submit the form after a short delay to allow the user to see the transcript
                setTimeout(() => {
                    document.getElementById('query-form').dispatchEvent(new Event('submit', { 'bubbles': true }));
                }, 1000);
            }
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error', event.error);
            document.getElementById('mic-status').textContent = '⚠️ Error: ' + event.error;
            stopListening();
        };
        
        recognition.onend = function() {
            stopListening();
        };
    }
    
    function toggleListening() {
        if (!recognition) {
            document.getElementById('mic-status').textContent = '❌ Speech recognition not supported';
            return;
        }
        
        if (isListening) {
            stopListening();
        } else {
            isListening = true;
            recognition.start();
        }
    }
    
    function stopListening() {
        if (isListening) {
            isListening = false;
            recognition.stop();
            document.getElementById('mic-status').textContent = '🎙️ Click to speak';
            document.getElementById('mic-button').classList.remove('listening');
        }
    }
    
    document.getElementById('mic-button').addEventListener('click', toggleListening);
    </script>
    """

# CSS for the microphone button and status
def get_css():
    return """
    <style>
    .mic-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    #mic-button {
        background-color: #f0f2f6;
        border: none;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.3s;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        margin-right: 10px;
    }
    #mic-button:hover {
        background-color: #e6e9ef;
    }
    #mic-button.listening {
        background-color: #ff4b4b;
        animation: pulse 1.5s infinite;
    }
    #mic-status {
        font-size: 14px;
        color: #555;
        margin-left: 10px;
    }
    #query-status {
        font-size: 14px;
        color: #555;
        margin-top: 5px;
        font-style: italic;
    }
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    </style>
    """

# HTML for the microphone button
def get_mic_button_html():
    return """
    <div class="mic-container">
        <button id="mic-button" type="button">🎙️</button>
        <span id="mic-status">🎙️ Click to speak</span>
    </div>
    <div id="query-status"></div>
    """

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot with Voice", layout="wide")
st.title("📄 PDF Chatbot with Voice & Gemini API 🤖")

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

# Add CSS and HTML components for voice recognition
st.markdown(get_css(), unsafe_allow_html=True)

# Create a form to properly handle the submission
with st.form(key="query_form", id="query-form"):
    # Add the microphone button
    st.markdown(get_mic_button_html(), unsafe_allow_html=True)
    
    # Add the query input field
    query = st.text_input("Ask a question from the PDF:", key="query-input", id="query-input")
    
    # Add the submit button
    submit_button = st.form_submit_button("Ask")

# Add the JavaScript after the form
st.markdown(get_speech_recognition_js(), unsafe_allow_html=True)

if submit_button and query and st.session_state.pdf_processed:
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

# Add instructions for using voice recognition
if st.session_state.pdf_processed:
    st.sidebar.title("Voice Recognition Instructions")
    st.sidebar.write("""
    ### How to use voice input:
    1. Click the microphone button 🎙️
    2. Speak your question clearly
    3. The system will automatically submit your question after you finish speaking
    4. If there's an error, try again or type your question manually
    
    **Note:** Voice recognition requires a microphone and works best in Chrome browsers.
    """)
