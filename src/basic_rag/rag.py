"""
Core RAG (Retrieval-Augmented Generation) pipeline for the basic RAG implementation.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

from dotenv import load_dotenv

# Handle both relative and absolute imports
try:
    from ..utils.chunking import get_document_chunks
except ImportError:
    # Fallback for direct script execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.chunking import get_document_chunks

# Load environment variables from .env file
load_dotenv()

class BasicRAGPipeline:
    """
    A basic RAG pipeline that loads a PDF, creates a vector store,
    and answers questions based on the document content.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize the RAG pipeline using OpenRouter.
        Models are loaded lazily when first needed.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided or found in environment variables.")

        self.base_url = base_url
        
        # Lazy loading - models will be initialized when first needed
        self._embeddings_model = None
        self._llm = None
        
        self.vector_store: Optional[FAISS] = None
        self.retriever = None

    @property
    def embeddings_model(self):
        """Lazy load the embeddings model when first accessed."""
        if self._embeddings_model is None:
            print("Loading embedding model (this may take a moment on first run)...")
            self._embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("Embedding model loaded successfully!")
        return self._embeddings_model
    
    @property
    def llm(self):
        """Lazy load the LLM when first accessed."""
        if self._llm is None:
            print("Initializing LLM connection...")
            self._llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model="deepseek/deepseek-r1-0528:free",
                default_headers={
                    "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8501"),  # Optional, for tracking
                    "X-Title": os.getenv("APP_NAME", "Basic RAG Pipeline")  # Optional, for tracking
                }
            )
            print("LLM initialized successfully!")
        return self._llm

    def setup_vector_store(self, pdf_path: Union[str, Path]) -> None:
        """
        Load a PDF, split it into chunks, and set up the FAISS vector store.

        Args:
            pdf_path (Union[str, Path]): Path to the PDF file.
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Use the existing chunking utility with recursive splitter
        chunks = get_document_chunks(
            file_path=pdf_path,
            splitter_type="recursive", 
            chunk_size=1000,
            chunk_overlap=200
        )

        if not chunks:
            raise ValueError("No chunks were created from the PDF. Check PDF content and chunking settings.")

        self.vector_store = FAISS.from_documents(chunks, self.embeddings_model)
        self.retriever = self.vector_store.as_retriever()
        print(f"Vector store set up successfully with {len(chunks)} chunks from {pdf_path}.")

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Helper function to format retrieved documents into a string.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def query(self, user_query: str) -> Tuple[str, List[Document]]:
        """
        Process a user query: retrieve relevant documents and generate an answer.

        Args:
            user_query (str): The user's question.

        Returns:
            Tuple[str, List[Document]]: A tuple containing the LLM's answer and the list of retrieved documents.
        
        Raises:
            RuntimeError: If the vector store has not been set up.
        """
        if not self.retriever:
            raise RuntimeError("Vector store not set up. Please call 'setup_vector_store' first with a PDF file.")

        template = """Answer the question based only on the following context:
            {context}

            Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Build the RAG chain
        rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        retrieved_docs = self.retriever.invoke(user_query)
        answer = rag_chain.invoke(user_query)

        return answer, retrieved_docs

    def get_retrieved_documents(self, user_query: str) -> List[Document]:
        """
        Only retrieves relevant documents for a given query without calling the LLM.

        Args:
            user_query (str): The user's question.

        Returns:
            List[Document]: A list of retrieved documents.

        Raises:
            RuntimeError: If the vector store has not been set up.
        """
        if not self.retriever:
            raise RuntimeError("Vector store not set up. Please call 'setup_vector_store' first with a PDF file.")
        
        return self.retriever.invoke(user_query)

# Example usage (optional, for direct script testing)
if __name__ == '__main__':
    try:
        if not os.path.exists(".env") and not os.getenv("OPENROUTER_API_KEY"):
            with open(".env", "w") as f:
                f.write("OPENROUTER_API_KEY=your_actual_openrouter_api_key_here\n")
            print("Created a dummy .env file. Please replace with your actual OpenRouter API key.")
            # Attempt to load again
            load_dotenv(override=True)

        # Ensure OPENROUTER_API_KEY is set
        if not os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") == "your_actual_openrouter_api_key_here":
            print("Please set your OPENROUTER_API_KEY in a .env file or as an environment variable to run this example.")
        else:
            rag_pipeline = BasicRAGPipeline()
            
            # Check if a PDF path is provided as an argument
            import sys
            if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
                pdf_path = Path(sys.argv[1])
                print(f"Using provided PDF: {pdf_path}")
            else:
                print("Error: No PDF file provided or file does not exist.")
                print("Usage: python -m src.basic_rag.rag path/to/your/pdf")
                sys.exit(1)
                
            rag_pipeline.setup_vector_store(pdf_path)
            
            test_query = "What is this document about?"
            print(f"\nQuery: {test_query}")
            answer, docs = rag_pipeline.query(test_query)
            print(f"Answer: {answer}")
            print(f"Retrieved {len(docs)} documents.")

    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")