"""
Vector Search Module for the Directory Chatbot

This module implements vector embedding and semantic search functionality,
enabling more natural language understanding and search capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.logging_utils import log_exceptions
from app.database.supabase_client import supabase, SupabaseDB

def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors using NumPy.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
    """
    # Ensure vectors are numpy arrays with correct shape
    v1 = np.asarray(vector1, dtype=np.float64)
    v2 = np.asarray(vector2, dtype=np.float64)
    
    # Handle zero vectors to prevent division by zero
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    # Calculate cosine similarity
    return np.dot(v1, v2) / (v1_norm * v2_norm)

def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate the cosine distance between two vectors using NumPy.
    This is 1 - cosine_similarity, to be compatible with the scipy implementation.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cosine distance (0.0 = identical, 1.0 = orthogonal, 2.0 = opposite)
    """
    return 1.0 - cosine_similarity(vector1, vector2)

# Configure logging
logger = logging.getLogger(__name__)

class VectorSearch:
    """
    Vector-based semantic search capabilities for the Directory Chatbot.
    
    This class provides methods for generating embeddings and performing
    semantic search operations on the knowledge base.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector search with a specific embedding model.
        
        Args:
            model_name: Name of the HuggingFace embedding model to use
        """
        self.model_name = model_name
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"Embedding model initialized: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_model = None
    
    @log_exceptions
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats, or None if failed
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return None
        
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
            
    @staticmethod
    def get_embedding_for_text(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[List[float]]:
        """
        Static method to get embedding for text without instantiating the full class.
        
        Args:
            text: Text to embed
            model_name: Name of the HuggingFace embedding model to use
            
        Returns:
            Embedding vector as a list of floats, or None if failed
        """
        try:
            model = HuggingFaceEmbeddings(model_name=model_name)
            embedding = model.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding with static method: {e}")
            return None
    
    @log_exceptions
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            Cosine similarity (1.0 = identical, 0.0 = unrelated)
        """
        try:
            # Use our NumPy-based implementation
            return cosine_similarity(vector1, vector2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    @log_exceptions
    def search_entities(self, 
                       query_text: str, 
                       entity_types: List[str] = None,
                       limit: int = 10) -> List[Dict]:
        """
        Search for entities similar to the query text.
        
        Args:
            query_text: Text to search for
            entity_types: Filter by entity types
            limit: Maximum number of results
            
        Returns:
            List of matching entities with similarity scores
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            # Generate embedding for search query
            query_embedding = self.embed_text(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Build SQL query
            sql = """
            SELECT *, 1 - (vector_embedding <=> $1) AS similarity
            FROM entities
            WHERE vector_embedding IS NOT NULL
            """
            
            params = [query_embedding]
            param_idx = 2
            
            # Add entity type filter if provided
            if entity_types and len(entity_types) > 0:
                sql += f" AND entity_type = ANY(${param_idx}::text[])"
                params.append(entity_types)
                param_idx += 1
            
            # Add ordering and limit
            sql += f" ORDER BY similarity DESC LIMIT ${param_idx}"
            params.append(limit)
            
            # Execute the query
            response = supabase.rpc("execute_sql", {
                "sql": sql,
                "params": params
            }).execute()
            
            if not response.data:
                return []
            
            # Format results
            results = []
            for entity in response.data:
                # Add similarity percentage for easier interpretation
                entity["similarity_percentage"] = round(entity.get("similarity", 0) * 100, 1)
                results.append(entity)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    @log_exceptions
    def contextual_search(self, 
                         query_text: str, 
                         context_text: str = None,
                         context_entity_id: str = None,
                         entity_types: List[str] = None,
                         limit: int = 10) -> List[Dict]:
        """
        Perform search with additional context to improve relevance.
        
        Args:
            query_text: Main query text
            context_text: Additional context text
            context_entity_id: ID of a related entity for context
            entity_types: Filter by entity types
            limit: Maximum number of results
            
        Returns:
            List of matching entities with similarity scores
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            # Combine query with context if provided
            combined_query = query_text
            
            if context_text:
                combined_query = f"{query_text} {context_text}"
            
            # If context entity provided, add its attributes
            if context_entity_id:
                entity = SupabaseDB.get_entity_by_id(context_entity_id)
                if entity and entity.get("attributes"):
                    # Extract relevant attributes
                    attr_text = ""
                    for key, value in entity.get("attributes", {}).items():
                        if isinstance(value, (str, int, float, bool)):
                            attr_text += f"{key}: {value}. "
                    
                    if attr_text:
                        combined_query = f"{combined_query} {attr_text}"
            
            # Now perform the search with the enhanced query
            return self.search_entities(
                query_text=combined_query,
                entity_types=entity_types,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return []
    
    @log_exceptions
    def relevant_chunk_search(self, 
                           query_text: str, 
                           text_chunks: List[str],
                           top_k: int = 3) -> List[Dict]:
        """
        Find the most relevant chunks from a list based on semantic similarity.
        
        Args:
            query_text: Query text
            text_chunks: List of text chunks to search through
            top_k: Number of top chunks to return
            
        Returns:
            List of dictionaries with chunk text and similarity scores
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embed_text(query_text)
            if not query_embedding:
                return []
            
            # Generate embeddings for each chunk
            chunk_embeddings = []
            for chunk in text_chunks:
                embedding = self.embed_text(chunk)
                if embedding:
                    chunk_embeddings.append((chunk, embedding))
            
            # Calculate similarities
            similarities = []
            for chunk, embedding in chunk_embeddings:
                similarity = self.cosine_similarity(query_embedding, embedding)
                similarities.append((chunk, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            results = []
            for chunk, similarity in similarities[:top_k]:
                results.append({
                    "text": chunk,
                    "similarity": similarity,
                    "similarity_percentage": round(similarity * 100, 1)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in relevant chunk search: {e}")
            return []

    @log_exceptions
    def update_entity_embedding(self, entity_id: str) -> bool:
        """
        Update the embedding vector for a specific entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return False
        
        try:
            # Get entity data
            entity = SupabaseDB.get_entity_by_id(entity_id)
            if not entity:
                logger.error(f"Entity not found: {entity_id}")
                return False
            
            # Prepare text for embedding
            entity_name = entity.get("entity_name", "")
            entity_type = entity.get("entity_type", "")
            
            text_to_embed = f"{entity_name} ({entity_type}). "
            
            # Add attributes to text
            for key, value in entity.get("attributes", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    text_to_embed += f"{key}: {value}. "
            
            # Generate embedding
            embedding = self.embed_text(text_to_embed)
            if not embedding:
                return False
            
            # Update entity with new embedding
            sql = """
            UPDATE entities
            SET vector_embedding = $1, updated_at = now()
            WHERE id = $2
            """
            
            response = supabase.rpc("execute_sql", {
                "sql": sql,
                "params": [embedding, entity_id]
            }).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating entity embedding: {e}")
            return False