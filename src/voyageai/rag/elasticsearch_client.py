"""
Elasticsearch BM25 Search Client

Provides keyword-based search using Elasticsearch's BM25 algorithm.
BM25 (Best Matching 25) is a bag-of-words ranking function that:
- Considers term frequency (TF)
- Applies inverse document frequency (IDF)
- Normalizes by document length

This complements vector search by:
- Finding exact keyword matches
- Handling specific terms the LLM might miss
- Being faster for simple queries

Architecture:
    Query → ES Analyzer → BM25 Scoring → Top-K Results

Requirements:
    - Elasticsearch 8.x running (docker-compose or cloud)
    - Index with travel knowledge documents

Usage:
    client = ElasticsearchClient()
    await client.initialize()
    results = await client.search("cherry blossom season Tokyo", top_k=10)
"""

import logging
from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

from voyageai.config import settings

logger = logging.getLogger(__name__)

# Default index settings for travel knowledge
TRAVEL_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "travel_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stemmer", "english_stop"]
                }
            },
            "filter": {
                "english_stemmer": {
                    "type": "stemmer",
                    "language": "english"
                },
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "travel_analyzer"
            },
            "title": {
                "type": "text",
                "analyzer": "travel_analyzer",
                "boost": 2.0  # Title matches weighted higher
            },
            "destination": {
                "type": "keyword"  # Exact match for filtering
            },
            "category": {
                "type": "keyword"
            },
            "source": {
                "type": "keyword"
            },
            "created_at": {
                "type": "date"
            }
        }
    }
}


class ElasticsearchClient:
    """
    Elasticsearch client for BM25 keyword search.
    
    Design Decisions:
    1. Async API for non-blocking operations
    2. Custom analyzer for travel-specific stemming
    3. Multi-match query across title and content
    4. Graceful degradation if ES is unavailable
    
    BM25 Parameters (ES defaults):
    - k1 = 1.2 (term frequency saturation)
    - b = 0.75 (document length normalization)
    """
    
    _instance: "ElasticsearchClient | None" = None
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "travel_knowledge"
    ):
        """
        Initialize Elasticsearch client.
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Default index to use
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self._client: AsyncElasticsearch | None = None
        self._available: bool = False
    
    @classmethod
    def get_instance(cls) -> "ElasticsearchClient":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ElasticsearchClient()
        return cls._instance
    
    async def initialize(self) -> bool:
        """
        Initialize Elasticsearch connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self._client is not None:
            return self._available
        
        try:
            logger.info(f"Connecting to Elasticsearch at {self.host}:{self.port}")
            
            self._client = AsyncElasticsearch(
                hosts=[{"host": self.host, "port": self.port, "scheme": "http"}],
                request_timeout=10,
            )
            
            # Test connection
            info = await self._client.info()
            logger.info(f"Connected to Elasticsearch {info['version']['number']}")
            
            # Create index if not exists
            await self._ensure_index()
            
            self._available = True
            return True
            
        except Exception as e:
            logger.warning(f"Elasticsearch not available: {e}")
            self._available = False
            return False
    
    async def _ensure_index(self) -> None:
        """Create the index if it doesn't exist."""
        if self._client is None:
            return
        
        try:
            exists = await self._client.indices.exists(index=self.index_name)
            if not exists:
                await self._client.indices.create(
                    index=self.index_name,
                    body=TRAVEL_INDEX_SETTINGS
                )
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                # Get document count
                count = await self._client.count(index=self.index_name)
                logger.info(f"Elasticsearch index '{self.index_name}' has {count['count']} documents")
        except Exception as e:
            logger.error(f"Failed to ensure index: {e}")
    
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """
        Add documents to the index.
        
        Args:
            documents: List of documents with content, title, metadata
            
        Returns:
            Number of documents indexed
        """
        if not self._available or self._client is None:
            logger.warning("Elasticsearch not available, skipping indexing")
            return 0
        
        from elasticsearch.helpers import async_bulk
        
        # Prepare bulk actions
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_source": {
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "destination": doc.get("destination", ""),
                    "category": doc.get("category", "general"),
                    "source": doc.get("source", "unknown"),
                }
            }
            if "id" in doc:
                action["_id"] = doc["id"]
            actions.append(action)
        
        # Bulk index
        success, failed = await async_bulk(self._client, actions)
        
        logger.info(f"Indexed {success} documents to Elasticsearch ({failed} failed)")
        return success
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        destination: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search using BM25 algorithm.
        
        Args:
            query: Search query
            top_k: Number of results
            destination: Optional filter by destination
            
        Returns:
            List of search results with content, metadata, and score
        """
        if not self._available or self._client is None:
            logger.debug("Elasticsearch not available, returning empty results")
            return []
        
        try:
            # Build query
            must_clause: dict[str, Any] = {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content", "destination"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",  # Handle typos
                }
            }
            
            # Add filter if destination specified
            filter_clause = None
            if destination:
                filter_clause = [{"term": {"destination": destination}}]
            
            body: dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": [must_clause],
                    }
                },
                "size": top_k,
                "_source": ["content", "title", "destination", "category", "source"],
            }
            
            if filter_clause:
                body["query"]["bool"]["filter"] = filter_clause
            
            # Execute search
            response = await self._client.search(
                index=self.index_name,
                body=body
            )
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "content": source.get("content", ""),
                    "metadata": {
                        "title": source.get("title", ""),
                        "destination": source.get("destination", ""),
                        "category": source.get("category", ""),
                        "source": source.get("source", ""),
                    },
                    "score": round(hit["_score"], 4),
                    "search_type": "bm25",
                })
            
            logger.info(f"BM25 search returned {len(results)} results for: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    async def delete_index(self) -> None:
        """Delete the index (for testing/reset)."""
        if not self._available or self._client is None:
            return
        
        try:
            await self._client.indices.delete(index=self.index_name)
            logger.warning(f"Deleted Elasticsearch index: {self.index_name}")
        except NotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
    
    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._available = False
    
    @property
    def is_available(self) -> bool:
        """Check if Elasticsearch is available."""
        return self._available


# Global singleton instance
es_client = ElasticsearchClient.get_instance()

