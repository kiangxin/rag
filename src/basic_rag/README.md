# Basic RAG (Retrieval-Augmented Generation) Pipeline

A simple and efficient RAG implementation that allows users to upload PDF documents and ask questions about their content using AI-powered question answering.

## Overview

This Basic RAG pipeline demonstrates the fundamental concepts of Retrieval-Augmented Generation:

1. **Document Processing**: Upload and process PDF documents
2. **Text Chunking**: Split documents into manageable chunks using recursive text splitter
3. **Vector Storage**: Store document embeddings using FAISS vector database
4. **Semantic Search**: Retrieve relevant document chunks based on user queries
5. **Answer Generation**: Generate contextual answers using LLM (DeepSeek R1) based on retrieved content

## Key Features

- 📄 **PDF Upload**: Simple drag-and-drop PDF upload interface
- 🔧 **Recursive Text Splitting**: Intelligent document chunking that preserves context
- 🚀 **FAISS Vector Database**: Fast and efficient similarity search
- 🤖 **DeepSeek R1 Integration**: Free AI model via OpenRouter for answer generation
- 🏃‍♂️ **Lazy Loading**: Optimized startup time with models loaded only when needed
- 💬 **Interactive Chat**: Ask questions and get answers based on document content
- 🔍 **Context Visibility**: View retrieved document chunks that informed the answer

## Architecture

```
PDF Document → Text Extraction → Recursive Chunking → Embeddings → FAISS Index
                                                                         ↓
User Query → Embedding → Similarity Search → Retrieved Chunks → LLM → Answer
```

### Components

- **Document Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` with 1000 character chunks and 200 character overlap
- **Embeddings**: Sentence-Transformers `all-MiniLM-L6-v2` model for creating vector representations
- **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient vector storage and retrieval
- **LLM**: DeepSeek R1 0528 (Free) via OpenRouter for answer generation
- **Frontend**: Streamlit web interface for user interaction

## Setup

### Prerequisites

- Python 3.8+
- OpenRouter API key (free tier available)

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd /path/to/rag/src/basic_rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r ../../requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   APP_URL=http://localhost:8501
   APP_NAME=Basic RAG Pipeline
   ```

## Usage

### Running the Streamlit App

```bash
# From the project root directory
cd /path/to/rag
streamlit run src/basic_rag/app.py
```

### Using the Pipeline Programmatically

```python
from src.basic_rag.rag import BasicRAGPipeline

# Initialize the pipeline
pipeline = BasicRAGPipeline()

# Set up vector store with your PDF
pipeline.setup_vector_store("path/to/your/document.pdf")

# Ask questions
answer, retrieved_docs = pipeline.query("What is this document about?")
print(f"Answer: {answer}")
```

### Running as a Module

```bash
# From the project root
python -m src.basic_rag.rag path/to/your/pdf
```

## File Structure

```
src/basic_rag/
├── README.md           # This file
├── app.py             # Streamlit web interface
├── rag.py             # Core RAG pipeline implementation
├── __init__.py        # Package initialization
└── data/
    └── temp_uploads/  # Temporary PDF storage
```

## Limitations

- **PDF Only**: Currently supports PDF documents only
- **Memory Usage**: Large documents may require significant memory for processing
- **Single Document**: One document at a time (no multi-document search)
- **Language**: Optimized for English text

## Future Enhancements

- Support for multiple document formats (DOCX, TXT, etc.)
- Multi-document search and comparison
- Advanced chunking strategies
- Conversation memory for follow-up questions
- Document preprocessing and cleaning
- Custom embedding models


