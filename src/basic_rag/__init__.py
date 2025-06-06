"""
Basic (Naive) RAG implementation.

This module provides a simple implementation of the Retrieval-Augmented Generation
approach where documents are retrieved based on query similarity and then used
to augment the prompt sent to an LLM.
"""

from .rag import BasicRAGPipeline

__all__ = ["BasicRAGPipeline"] 