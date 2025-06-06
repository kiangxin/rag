"""
Streamlit frontend for the Basic RAG Pipeline.

Allows users to upload a PDF, ask questions, and get answers based on the document content.
"""
import streamlit as st
import os
import torch

from pathlib import Path
from dotenv import load_dotenv

from rag import BasicRAGPipeline

# Load environment variables from .env file, if it exists
load_dotenv()
torch.classes.__path__ = []

# --- Page Configuration ---
st.set_page_config(
    page_title="Basic RAG with PDF",
    page_icon="📄",
    layout="wide"
)

# --- Helper Functions ---
def initialize_rag_pipeline() -> BasicRAGPipeline:
    """
    Initialize or retrieve the RAG pipeline from session state.
    Only creates the pipeline object without loading heavy models.
    """
    if "rag_pipeline" not in st.session_state or st.session_state.rag_pipeline is None:
        try:
            # Load API key from environment variables
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                st.error("OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")
                return None
            
            # Create pipeline object (models will be lazy loaded)
            st.session_state.rag_pipeline = BasicRAGPipeline(api_key=api_key)
            st.success("RAG Pipeline initialized (models will load when needed)!")
        except ValueError as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            return None
    return st.session_state.rag_pipeline

# --- Main Application UI ---
st.title("📄 Basic RAG QA with PDF Document")
st.markdown("Upload a PDF document, and then ask questions based on its content.")

# --- Sidebar for API Key and PDF Upload ---
with st.sidebar:
    st.header("Configuration")

    # Check if API key is available
    st.subheader("OpenRouter API Configuration")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        st.success("✅ OpenRouter API Key loaded from environment")
    else:
        st.error("❌ OPENROUTER_API_KEY not found in environment variables")
        st.info("Please add your OpenRouter API key to the .env file in the project root.")
        st.stop()
    
    # Model Information
    st.subheader("Model Information")
    st.info("Using DeepSeek R1 0528 (Free) for text generation")
    st.info("Using Sentence-Transformers MiniLM-L6-v2 for embeddings")

    # PDF File Uploader
    st.subheader("Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="The content of this PDF will be used to answer your questions."
    )

    if uploaded_file is not None:
        if st.button("Process PDF", key="process_pdf_button"):
            # Initialize RAG pipeline only when needed
            with st.spinner("Initializing RAG pipeline..."):
                rag_pipeline = initialize_rag_pipeline()
                if not rag_pipeline:
                    st.stop()
            
            # Save uploaded file temporarily to pass its path
            temp_dir = Path("data/temp_uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_pdf_path = temp_dir / uploaded_file.name

            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing PDF and building vector store..."):
                try:
                    rag_pipeline.setup_vector_store(temp_pdf_path)
                    st.session_state.pdf_processed = True
                    st.session_state.processed_pdf_name = uploaded_file.name
                    st.success(f"'{uploaded_file.name}' processed and ready for questions!")
                except FileNotFoundError:
                    st.error("Error: Temporary PDF file not found after upload. Please try again.")
                    st.session_state.pdf_processed = False
                except ValueError as e:
                    st.error(f"Error processing PDF: {e}")
                    st.session_state.pdf_processed = False
                except Exception as e:
                    st.error(f"An unexpected error occurred during PDF processing: {e}")
                    st.session_state.pdf_processed = False
                finally:
                    # Clean up the temporary file
                    if temp_pdf_path.exists():
                        temp_pdf_path.unlink()
    
    if st.session_state.get("pdf_processed", False):
        st.info(f"Currently using: **{st.session_state.get('processed_pdf_name', '')}**")

# --- Main Interaction Area --- 
if not st.session_state.get("pdf_processed", False):
    st.info("Please upload and process a PDF document using the sidebar to begin.")
else:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question about the document:", key="query_input")

    if user_query:
        if st.button("Get Answer", key="get_answer_button"):
            # Get the pipeline from session state
            rag_pipeline = st.session_state.get("rag_pipeline")
            if not rag_pipeline:
                st.error("Pipeline not initialized. Please process a PDF first.")
                st.stop()
                
            with st.spinner("Searching for answer using DeepSeek R1..."):
                try:
                    answer, retrieved_docs = rag_pipeline.query(user_query)
                    
                    st.subheader("Answer:")
                    st.markdown(answer)
                    
                    with st.expander("Show Retrieved Context"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Context Chunk {i+1}:**")
                            st.caption(doc.page_content)
                except RuntimeError as e:
                    st.error(f"Runtime Error: {e}. Has the PDF been processed?")
                except Exception as e:
                    st.error(f"An error occurred while querying: {e}")
