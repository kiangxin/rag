"""
PDF document loading and text chunking utilities.

This module provides functions for loading PDF documents and splitting them into chunks
using different text splitters from LangChain.
"""
import pymupdf4llm
import os

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain.schema import Document


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """
    Load a PDF document using pymupdf4llm and convert to Document object.

    Args:
        file_path (Union[str, Path]): Path to the PDF file.

    Returns:
        List[Document]: List containing a single Document object with the PDF content.
    """
    # Convert to string path if it's a Path object
    file_path_str = str(file_path)
    
    # Extract the PDF content as markdown
    md_text = pymupdf4llm.to_markdown(file_path_str)
    
    # Create metadata
    metadata = {
        "source": file_path_str,
        "file_name": os.path.basename(file_path_str)
    }
    
    # Create a Document object
    document = Document(page_content=md_text, metadata=metadata)
    
    # Return as a list to maintain compatibility with the rest of the code
    return [document]


def split_with_character_text_splitter(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n",
) -> List[Document]:
    """
    Split documents using CharacterTextSplitter.

    Args:
        documents (List[Document]): List of documents to split.
        chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
        separator (str, optional): Separator to use for splitting. Defaults to "\n".

    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)


def split_with_recursive_character_text_splitter(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", " ", ""],
) -> List[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter.
    
    This splitter is more sophisticated than CharacterTextSplitter as it tries to split
    on multiple separators in order of preference.

    Args:
        documents (List[Document]): List of documents to split.
        chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
        separators (List[str], optional): List of separators to use for splitting, 
                                         in order of preference. 
                                         Defaults to ["\n\n", "\n", " ", ""].

    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)


def split_with_markdown_text_splitter(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents using MarkdownTextSplitter.
    
    This splitter is optimized for Markdown content and respects Markdown structure.

    Args:
        documents (List[Document]): List of documents to split.
        chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return text_splitter.split_documents(documents)


def get_document_chunks(
    file_path: Union[str, Path],
    splitter_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> List[Document]:
    """
    Load a PDF and split it into chunks using the specified splitter.

    Args:
        file_path (Union[str, Path]): Path to the PDF file.
        splitter_type (str, optional): Type of splitter to use. 
                                      Options: "character", "recursive", "markdown".
                                      Defaults to "recursive".
        chunk_size (int, optional): Maximum size of chunks. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.
        **kwargs: Additional arguments to pass to the splitter.

    Returns:
        List[Document]: List of document chunks.
        
    Raises:
        ValueError: If an invalid splitter_type is provided.
    """
    documents = load_pdf(file_path)
    
    if splitter_type == "character":
        return split_with_character_text_splitter(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separator=kwargs.get("separator", "\n"),
        )
    elif splitter_type == "recursive":
        return split_with_recursive_character_text_splitter(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators", ["\n\n", "\n", " ", ""]),
        )
    elif splitter_type == "markdown":
        return split_with_markdown_text_splitter(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(
            f"Invalid splitter_type: {splitter_type}. "
            "Choose from 'character', 'recursive', or 'markdown'."
        )
