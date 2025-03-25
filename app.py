import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import time
import google.generativeai as genai

# Hugging Face Transformers for Image Captioning
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# For text embedding and similarity search
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# Configure page
st.set_page_config(page_title="PDF Q&A with Image Captioning", page_icon="📄", layout="wide")

class ImageCaptioner:
    def __init__(self):
        """Initialize image captioning model"""
        with st.spinner("Loading Image Captioning Model..."):
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Set generation parameters
            self.model.config.max_length = 16
            self.model.config.no_repeat_ngram_size = 3
            self.model.config.num_beams = 4

    def generate_caption(self, image_path):
        """Generate caption for a given image"""
        try:
            # Open and preprocess image
            image = Image.open(image_path)
            
            # Prepare image for model
            pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate caption
            output_ids = self.model.generate(pixel_values)
            
            # Decode the generated caption
            caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            return "Unable to generate caption"

class PDFProcessor:
    def __init__(self):
        # Initialize AI models
        with st.spinner("Loading AI models... this may take a moment"):
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.image_captioner = ImageCaptioner()
        
        # Storage for PDF content
        self.image_dict = {}
        self.image_captions = {}

    def reset(self):
        """Reset the processor state for a new PDF"""
        try:
            # Reset dictionaries
            self.image_dict.clear()
            self.image_captions.clear()
            
            # Reinitialize image captioner if needed
            if not hasattr(self, 'image_captioner'):
                self.image_captioner = ImageCaptioner()
            
            st.success("Processor reset successfully!")
        except Exception as e:
            st.error(f"Error during reset: {str(e)}")

    def _rectangles_overlap_horizontally(self, rect1, rect2):
        """Check if two rectangles overlap horizontally"""
        x0_1, _, x1_1, _ = rect1
        x0_2, _, x1_2, _ = rect2
        return max(x0_1, x0_2) <= min(x1_1, x1_2)

    def _rectangles_overlap_vertically(self, rect1, rect2):
        """Check if two rectangles overlap vertically"""
        _, y0_1, _, y1_1 = rect1
        _, y0_2, _, y1_2 = rect2
        return max(y0_1, y0_2) <= min(y1_1, y1_2)

    def _find_text_near_image(self, text_blocks, img_rect, margin=100):
        """Find text surrounding the image (above, below, left, right)"""
        x0, y0, x1, y1 = img_rect
        
        above_text = []
        below_text = []
        left_text = []
        right_text = []
        
        for text, bbox in text_blocks:
            tx0, ty0, tx1, ty1 = bbox
            
            # Above the image
            if ty1 < y0 and ty1 > y0 - margin and self._rectangles_overlap_horizontally(img_rect, bbox):
                above_text.append(text)
            
            # Below the image
            if ty0 > y1 and ty0 < y1 + margin and self._rectangles_overlap_horizontally(img_rect, bbox):
                below_text.append(text)
            
            # Left of the image
            if tx1 < x0 and tx1 > x0 - margin and self._rectangles_overlap_vertically(img_rect, bbox):
                left_text.append(text)
            
            # Right of the image
            if tx0 > x1 and tx0 < x1 + margin and self._rectangles_overlap_vertically(img_rect, bbox):
                right_text.append(text)
        
        # Combine all context text
        context = []
        if above_text:
            context.append("Above: " + " ".join(above_text))
        if below_text:
            context.append("Below: " + " ".join(below_text))
        if left_text:
            context.append("Left: " + " ".join(left_text))
        if right_text:
            context.append("Right: " + " ".join(right_text))
        
        return " | ".join(context)

    def extract_text_and_images_from_pdf(self, pdf_path):
        """Extract text and images from PDF with positional information"""
        doc = fitz.open(pdf_path)
        text_per_page = []
        images_per_page = {}
        image_context_dict = {}
        
        # Create progress bar
        progress_bar = st.progress(0)
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            # Update progress bar
            progress_bar.progress((page_num + 1) / total_pages)
            
            # Add a status message for current page
            st.info(f"Processing Page {page_num + 1}/{total_pages}")
            
            page = doc[page_num]
            text = page.get_text("text")
            text_per_page.append((page_num, text))
            
            # Get blocks with their bounding boxes (includes text and images)
            blocks = page.get_text("dict")["blocks"]
            text_blocks = []
            
            # Extract text blocks with their positions
            for b in blocks:
                if b["type"] == 0:  # Type 0 is text
                    block_text = ""
                    for line in b["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    if block_text.strip():
                        # Store text with its bounding box (x0, y0, x1, y1)
                        text_blocks.append((block_text.strip(), b["bbox"]))
            
            # Process images
            images = page.get_images(full=True)
            images_per_page[page_num] = []
            
            for img_idx, img in enumerate(images):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image position on page
                    img_rect = None
                    for block in blocks:
                        if block.get("type") == 1:  # Type 1 is image
                            if block.get("xref") == xref:
                                img_rect = block["bbox"]  # [x0, y0, x1, y1]
                                break
                    
                    # If we couldn't find image position in blocks, try another method
                    if img_rect is None:
                        try:
                            img_rect = [item for item in page.get_image_info() if item["xref"] == xref][0]["bbox"]
                        except (IndexError, KeyError):
                            img_rect = [0, 0, page.rect.width, page.rect.height]
                    
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        tmp_img.write(image_bytes)
                        img_path = tmp_img.name
                    
                    # Find text near the image (above, below, left, right)
                    context_text = self._find_text_near_image(text_blocks, img_rect)
                    
                    # Generate image caption
                    image_caption = self.image_captioner.generate_caption(img_path)
                    
                    # Store image with its context and caption
                    image_context_dict[img_path] = {
                        'page': page_num,
                        'position': img_rect,
                        'context_text': context_text,
                        'caption': image_caption
                    }
                    
                    # Store in the images list
                    images_per_page[page_num].append({
                        'path': img_path,
                        'context': context_text,
                        'caption': image_caption
                    })
                    
                    # Add to image dictionary - using context as key
                    if context_text.strip():
                        self.image_dict[context_text] = img_path
                        self.image_captions[context_text] = image_caption
                except Exception as e:
                    st.error(f"Error processing image on page {page_num+1}: {str(e)}")
                    continue
        
        # Clear progress bar and status message
        progress_bar.empty()
        
        doc.close()
        return text_per_page, images_per_page, image_context_dict

    def search_images_by_text(self, query):
        """Search for images based on text query"""
        # Encode query text
        query_embedding = self.text_model.encode(query)
        
        # Calculate similarities for text keys
        similarities = {}
        for key in self.image_dict.keys():
            key_embedding = self.text_model.encode(key)
            similarity = np.dot(query_embedding, key_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(key_embedding)
            )
            similarities[key] = similarity
        
        # Return top matches
        top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        result_images = [self.image_dict[key] for key, score in top_matches if score > 0.4]
        
        return result_images, top_matches

def create_vector_store(text_per_page):
    """Create a FAISS vector store from PDF text"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"page": page_num, "chunk": i}
            )
            documents.append(doc)
    
    return FAISS.from_documents(documents, embeddings)

def query_gemini(prompt, context):
    """Query Gemini API with context and prompt"""
    # Ensure API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key is not set. Please set the GOOGLE_API_KEY environment variable.")
        return "API key error: Unable to generate response."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Context: {context}\nUser Query: {prompt}\nProvide a concise answer based on the context."
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}\n\nBased on the context, here's what I can tell you:\n{context[:500]}..."

def main():
    st.title("📄 PDF Q&A Assistant with Image Captioning")
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = PDFProcessor()
    
    # Ensure all necessary session state variables exist
    session_vars = [
        'pdf_processed', 'text_per_page', 'images_per_page', 
        'image_context_dict', 'vector_store', 'current_file_name'
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Check if we need to process a new PDF
        if st.session_state.current_file_name != uploaded_file.name:
            # Reset state for new PDF
            st.session_state.pdf_processed = False
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.processor.reset()
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Process the PDF if not already processed
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                start_time = time.time()
                st.session_state.text_per_page, st.session_state.images_per_page, st.session_state.image_context_dict = st.session_state.processor.extract_text_and_images_from_pdf(pdf_path)
                st.session_state.vector_store = create_vector_store(st.session_state.text_per_page)
                st.session_state.pdf_processed = True
                processing_time = time.time() - start_time
                st.success(f"PDF processed in {processing_time:.2f} seconds!")
        
        # Tabs for different functionalities
        tab1, tab2 = st.tabs(["Q&A", "Image Captions"])
        
        with tab1:
            # Question answering section
            st.header("Ask Questions")
            query = st.text_input("Enter your question:")
            
            if query:
                with st.spinner("Searching for answer..."):
                    # Search for relevant text
                    docs = st.session_state.vector_store.similarity_search(query, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Find relevant images
                    relevant_images, top_matches = st.session_state.processor.search_images_by_text(query)
                    
                    # Generate answer
                    answer = query_gemini(query, context)
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display relevant images BELOW the answer
                    if relevant_images:
                        st.subheader("Relevant Images")
                        columns = st.columns(min(len(relevant_images), 3))
                        
                        for i, img_path in enumerate(relevant_images):
                            with columns[i % len(columns)]:
                                st.image(img_path, use_column_width=True)
                                # Display image caption if available
                                for context, path in st.session_state.processor.image_dict.items():
                                    if path == img_path and context in st.session_state.processor.image_captions:
                                        st.caption(f"Caption: {st.session_state.processor.image_captions[context]}")
                    
                    # Display page numbers for reference
                    page_nums = set(doc.metadata['page'] for doc in docs)
                    if page_nums:
                        page_list = [str(p + 1) for p in page_nums]
                        st.caption(f"Information found on page(s): {', '.join(page_list)}")
        
        with tab2:
            # Image Captions section
            st.header("Image Captions")
            if st.session_state.images_per_page:
                for page_num, images in st.session_state.images_per_page.items():
                    st.subheader(f"Page {page_num + 1}")
                    columns = st.columns(min(len(images), 3))
                    
                    for i, img_info in enumerate(images):
                        with columns[i % len(columns)]:
                            st.image(img_info['path'], use_column_width=True)
                            st.caption(f"Caption: {img_info.get('caption', 'No caption generated')}")
                            st.caption(f"Context: {img_info.get('context', 'No context')}")
            else:
                st.info("No images found in the PDF")
    else:
        st.info("👆 Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()
