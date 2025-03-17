# app.py
 # Standard library imports
 import os
 import re
 import tempfile
 
 # Third-party imports
 import streamlit as st
 @@ -15,9 +15,8 @@
 from langchain_community.vectorstores import FAISS
 from langchain_community.embeddings import HuggingFaceEmbeddings
 from langchain.text_splitter import CharacterTextSplitter
 from langchain_core.documents import Document
 from langchain.docstore.document import Document
 from sentence_transformers import SentenceTransformer
 from wordcloud import WordCloud
 
 # Configure API with environment variable
 api_key = os.getenv("GOOGLE_API_KEY")
 @@ -30,13 +29,7 @@
 embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
 embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
 # Function to clean text
 def clean_text(text):
     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
     return text
 
 # Function to extract text and images from PDF (now stores image bytes)
 # Function to extract text and images from PDF
 def extract_text_and_images_from_pdf(pdf_path):
     doc = fitz.open(pdf_path)
     text_per_page = []
 @@ -51,7 +44,9 @@ def extract_text_and_images_from_pdf(pdf_path):
             xref = img[0]
             base_image = doc.extract_image(xref)
             image_bytes = base_image["image"]
             images_per_page[page_num].append(image_bytes)  # Store bytes instead of file paths
             with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                 tmp_img.write(image_bytes)
                 images_per_page[page_num].append(tmp_img.name)
     doc.close()
     return text_per_page, images_per_page
 
 @@ -67,31 +62,18 @@ def index_pdf_text(text_per_page):
     vector_store = FAISS.from_documents(documents, embedding_function)
     return vector_store
 
 # Function to query Gemini API with concise text prompt (using gemini-2.0-pro)
 # Function to query Gemini API with concise prompt
 def query_gemini(prompt, context):
     try:
         model = genai.GenerativeModel("gemini-2.0-pro")
         model = genai.GenerativeModel("gemini-2.0-flash")
         response = model.generate_content(
             f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise text answer suitable for exam preparation."
             f"Context: {context}\nUser Query: {prompt}\nProvide a short and concise answer suitable for exam preparation."
         )
         return response.text
     except Exception as e:
         return f"Error querying Gemini API: {str(e)}"
 
 # Function to generate an image based on a prompt (using gemini-2.0-pro)
 def generate_image(prompt):
     try:
         model = genai.GenerativeModel("gemini-2.0-pro")
         response = model.generate_content(f"{prompt}\nGenerate an image related to the answer.")
         for part in response.parts:  # Check response parts for image data
             if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith("image/"):
                 return part.inline_data.data  # Return image bytes if found
         return None  # Return None if no image is generated
     except Exception as e:
         print(f"Error generating image: {str(e)}")
         return None
 
 # Function to search PDF and answer with images and word cloud
 # Function to search PDF and answer with images
 def search_pdf_and_answer(query, vector_store, images_per_page):
     docs = vector_store.similarity_search(query, k=3)
     context = "\n".join([doc.page_content for doc in docs])
 @@ -100,47 +82,26 @@ def search_pdf_and_answer(query, vector_store, images_per_page):
     relevant_images = []
     for page_num in page_nums:
         relevant_images.extend(images_per_page.get(page_num, []))
 
     # Generate word cloud
     clean_context = clean_text(context)
     wordcloud = WordCloud().generate(clean_context)
     wordcloud_image = wordcloud.to_image()
 
     # Generate new image based on the answer
     generated_image_data = generate_image(f"Answer: {answer}")
 
     return answer, relevant_images, wordcloud_image, generated_image_data
     return answer, relevant_images
 
 # Streamlit UI
 st.title("📄 PDF Chatbot with Gemini API and Visual Aids 🤖")
 st.title("📄 PDF Chatbot with Gemini API 🤖")
 uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
 
 if uploaded_file:
     with st.spinner("Processing PDF... Please wait..."):
         with open("temp.pdf", "wb") as f:
             f.write(uploaded_file.read())
         text_per_page, images_per_page = extract_text_and_images_from_pdf("temp.pdf")
         vector_store = index_pdf_text(text_per_page)
         os.remove("temp.pdf")  # Clean up temporary file
     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
         tmp_file.write(uploaded_file.read())
         temp_path = tmp_file.name
     st.info("Processing PDF... Please wait...")
     text_per_page, images_per_page = extract_text_and_images_from_pdf(temp_path)
     vector_store = index_pdf_text(text_per_page)
     st.success("PDF successfully indexed! ✅")
     query = st.text_input("Ask a question from the PDF:")
 
     if query:
         with st.spinner("Generating response..."):
             answer, relevant_images, wordcloud_image, generated_image_data = search_pdf_and_answer(query, vector_store, images_per_page)
         
         st.write("### 🤖 Answer")
         answer, relevant_images = search_pdf_and_answer(query, vector_store, images_per_page)
         st.write("### 🤖 Answer:")
         st.write(answer)
         
         if relevant_images:
             st.write("#### Relevant Images from PDF")
             for img_bytes in relevant_images:
                 st.image(img_bytes, use_column_width=True)
         
         if generated_image_data:
             st.write("#### Generated Image Based on Answer")
             st.image(generated_image_data, use_column_width=True)
         
         st.write("#### Word Cloud")
         st.image(wordcloud_image)
 
             st.write("#### Relevant Images:")
             for img_path in relevant_images:
                 st.image(img_path, use_column_width=True)
