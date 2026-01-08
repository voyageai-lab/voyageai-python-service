"""
ChromaDB Vector Search Client

Provides semantic (vector) search using ChromaDB with OpenAI embeddings.
ChromaDB is a lightweight, in-process vector database that:
- Stores embeddings with metadata
- Supports similarity search (cosine, L2, IP)
- Runs embedded (no external server needed)

Architecture:
    Query → OpenAI Embedding → ChromaDB Query → Top-K Similar Docs

Usage:
    client = ChromaClient()
    await client.initialize()
    results = await client.search("best time to visit Tokyo", top_k=10)
"""

import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI

from voyageai.config import settings

logger = logging.getLogger(__name__)

# Embedding model (OpenAI's latest small model - cost effective)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


class ChromaClient:
    """
    ChromaDB client for vector search.
    
    Design Decisions:
    1. Uses persistent storage (not ephemeral) for production use
    2. OpenAI embeddings for high-quality semantic understanding
    3. Async API for non-blocking operations
    4. Singleton pattern to reuse connections
    
    Collections:
    - travel_knowledge: Travel guides, tips, destination info
    - tool_metadata: Tool descriptions for Tool-RAG (Module 10)
    """
    
    _instance: "ChromaClient | None" = None
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "travel_knowledge"
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Where to store the database
            collection_name: Default collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._openai: AsyncOpenAI | None = None
    
    @classmethod
    def get_instance(cls) -> "ChromaClient":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ChromaClient()
        return cls._instance
    
    async def initialize(self) -> None:
        """
        Initialize ChromaDB and OpenAI clients.
        
        This should be called during application startup.
        """
        if self._client is not None:
            return
        
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        
        # Create persistent client
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Travel knowledge for RAG",
                "hnsw:space": "cosine",  # Cosine similarity
            }
        )
        
        # Initialize OpenAI for embeddings
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        
        count = self._collection.count()
        logger.info(f"ChromaDB initialized: {count} documents in '{self.collection_name}'")
    
    async def _get_embedding(self, text: str) -> list[float]:
        """
        Get embedding vector for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1536 dimensions)
        """
        if self._openai is None:
            raise RuntimeError("ChromaClient not initialized. Call initialize() first.")
        
        response = await self._openai.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    async def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs (generated if not provided)
            
        Returns:
            Number of documents added
        """
        if self._collection is None:
            raise RuntimeError("ChromaClient not initialized")
        
        # Generate IDs if not provided
        if ids is None:
            import hashlib
            ids = [
                hashlib.md5(doc.encode()).hexdigest()[:16]
                for doc in documents
            ]
        
        # Get embeddings for all documents
        embeddings = []
        for doc in documents:
            embedding = await self._get_embedding(doc)
            embeddings.append(embedding)
        
        # Add to collection
        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids,
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return len(documents)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of search results with content, metadata, and score
        """
        if self._collection is None:
            raise RuntimeError("ChromaClient not initialized")
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                # Convert distance to similarity score
                # ChromaDB returns L2 distance, convert to 0-1 score
                distance = results["distances"][0][i] if results.get("distances") else 0
                # For cosine distance, score = 1 - distance/2
                similarity = 1 - (distance / 2)
                
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "score": round(similarity, 4),
                    "distance": round(distance, 4),
                    "search_type": "vector",
                })
        
        logger.info(f"Vector search returned {len(formatted)} results for: {query[:50]}...")
        return formatted
    
    async def delete_collection(self) -> None:
        """Delete the current collection (for testing/reset)."""
        if self._client is None:
            return
        
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def count(self) -> int:
        """Get document count in collection."""
        if self._collection is None:
            return 0
        return self._collection.count()


# Global singleton instance
chroma_client = ChromaClient.get_instance()

