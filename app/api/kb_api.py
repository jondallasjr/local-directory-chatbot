"""
Knowledge Base API endpoints for the Directory Chatbot.

This module provides API endpoints for managing the graph-based knowledge base,
including entity and relationship management, and semantic search capabilities.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
import logging
import json

from app.api.api import verify_api_key
from app.database.graph_kb import GraphKnowledgeBase
from app.models.vector_search import VectorSearch
from app.utils.logging_utils import log_exceptions

# Configure logging
logger = logging.getLogger(__name__)

# Initialize knowledge base and vector search
kb = GraphKnowledgeBase()
vector_search = VectorSearch()

# Create API router
router = APIRouter(
    prefix="/api/kb",
    tags=["knowledge-base"],
    dependencies=[Depends(verify_api_key)]
)

# Entity Management Endpoints
@router.post("/entities")
@log_exceptions
async def create_entity(
    entity_type: str = Body(..., description="Type of entity"),
    entity_name: str = Body(..., description="Name of the entity"),
    attributes: Dict = Body({}, description="Entity attributes"),
    created_by: Optional[str] = Body(None, description="ID of the creator"),
    tags: List[str] = Body([], description="Entity tags"),
    generate_embedding: bool = Body(True, description="Whether to generate an embedding")
):
    """Create a new entity in the knowledge base."""
    try:
        entity_id = kb.add_entity(
            entity_type=entity_type,
            entity_name=entity_name,
            attributes=attributes,
            created_by=created_by,
            tags=tags,
            generate_embedding=generate_embedding
        )
        
        if entity_id:
            return {"success": True, "entity_id": entity_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to create entity")
    
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}")
@log_exceptions
async def get_entity(entity_id: str):
    """Get entity details by ID."""
    try:
        entity = kb.get_entity(entity_id)
        
        if entity:
            return entity
        else:
            raise HTTPException(status_code=404, detail="Entity not found")
    
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/entities/{entity_id}")
@log_exceptions
async def update_entity(
    entity_id: str,
    entity_type: Optional[str] = Body(None, description="Type of entity"),
    entity_name: Optional[str] = Body(None, description="Name of the entity"),
    attributes: Optional[Dict] = Body(None, description="Entity attributes"),
    tags: Optional[List[str]] = Body(None, description="Entity tags"),
    update_embedding: bool = Body(False, description="Whether to update the embedding")
):
    """Update an existing entity."""
    try:
        success = kb.update_entity(
            entity_id=entity_id,
            entity_type=entity_type,
            entity_name=entity_name,
            attributes=attributes,
            tags=tags,
            update_embedding=update_embedding
        )
        
        if success:
            return {"success": True, "entity_id": entity_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to update entity")
    
    except Exception as e:
        logger.error(f"Error updating entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity_id}")
@log_exceptions
async def delete_entity(entity_id: str):
    """Delete an entity from the knowledge base."""
    try:
        success = kb.delete_entity(entity_id)
        
        if success:
            return {"success": True, "entity_id": entity_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete entity")
    
    except Exception as e:
        logger.error(f"Error deleting entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Relationship Management Endpoints
@router.post("/relationships")
@log_exceptions
async def create_relationship(
    from_entity_id: str = Body(..., description="ID of the source entity"),
    to_entity_id: str = Body(..., description="ID of the target entity"),
    relationship_type: str = Body(..., description="Type of relationship"),
    attributes: Dict = Body({}, description="Relationship attributes"),
    weight: float = Body(1.0, description="Relationship weight (0.0-1.0)")
):
    """Create a relationship between two entities."""
    try:
        relationship_id = kb.add_relationship(
            from_entity_id=from_entity_id,
            to_entity_id=to_entity_id,
            relationship_type=relationship_type,
            attributes=attributes,
            weight=weight
        )
        
        if relationship_id:
            return {"success": True, "relationship_id": relationship_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to create relationship")
    
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/relationships")
@log_exceptions
async def delete_relationship(
    from_entity_id: str = Query(..., description="ID of the source entity"),
    to_entity_id: str = Query(..., description="ID of the target entity"),
    relationship_type: Optional[str] = Query(None, description="Type of relationship")
):
    """Delete a relationship between two entities."""
    try:
        success = kb.delete_relationship(
            from_entity_id=from_entity_id,
            to_entity_id=to_entity_id,
            relationship_type=relationship_type
        )
        
        if success:
            return {"success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete relationship")
    
    except Exception as e:
        logger.error(f"Error deleting relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/relationships/{entity_id}")
@log_exceptions
async def get_entity_relationships(
    entity_id: str,
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    direction: str = Query("both", description="Relationship direction: outgoing, incoming, or both")
):
    """Get relationships for an entity."""
    try:
        relationships = kb.get_relationships(
            entity_id=entity_id,
            relationship_type=relationship_type,
            direction=direction
        )
        
        return {"entity_id": entity_id, "relationships": relationships, "count": len(relationships)}
    
    except Exception as e:
        logger.error(f"Error getting relationships for entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search Endpoints
@router.get("/search")
@log_exceptions
async def search_entities(
    search_text: str = Query(..., description="Text to search for"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags to filter by"),
    limit: int = Query(10, description="Maximum number of results to return"),
    use_vector_search: bool = Query(True, description="Whether to use vector search")
):
    """Search for entities based on text, type, and tags."""
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        # Perform search
        results = kb.search_entities(
            search_text=search_text,
            entity_type=entity_type,
            tags=tag_list,
            limit=limit,
            use_vector_search=use_vector_search
        )
        
        return {"search_text": search_text, "results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/traversal/{entity_id}")
@log_exceptions
async def traversal_search(
    entity_id: str,
    max_depth: int = Query(2, description="Maximum traversal depth"),
    max_breadth: int = Query(5, description="Maximum breadth per entity"),
    relationship_types: Optional[str] = Query(None, description="Comma-separated list of relationship types")
):
    """Perform a graph traversal from a starting entity."""
    try:
        # Parse relationship types if provided
        rel_types = None
        if relationship_types:
            rel_types = [rel.strip() for rel in relationship_types.split(",")]
        
        # Perform traversal
        result = kb.traversal_search(
            start_entity_id=entity_id,
            max_depth=max_depth,
            max_breadth=max_breadth,
            relationship_types=rel_types
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error in traversal search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hybrid-search")
@log_exceptions
async def hybrid_search(
    search_text: str = Query(..., description="Text to search for"),
    entity_types: Optional[str] = Query(None, description="Comma-separated list of entity types"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags"),
    expand_results: bool = Query(True, description="Whether to expand results with related entities"),
    max_expansion_depth: int = Query(1, description="Maximum depth for expanding results"),
    limit: int = Query(10, description="Maximum number of direct search results")
):
    """Perform a hybrid search combining vector similarity with graph expansion."""
    try:
        # Parse entity types and tags if provided
        entity_type_list = None
        if entity_types:
            entity_type_list = [et.strip() for et in entity_types.split(",")]
        
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        # Perform hybrid search
        results = kb.hybrid_search(
            search_text=search_text,
            entity_types=entity_type_list,
            tags=tag_list,
            expand_results=expand_results,
            max_expansion_depth=max_expansion_depth,
            limit=limit
        )
        
        return results
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector Search Endpoints
@router.post("/vector-search")
@log_exceptions
async def vector_search_endpoint(
    query_text: str = Body(..., description="Text to search for"),
    entity_types: List[str] = Body([], description="Filter by entity types"),
    limit: int = Body(10, description="Maximum number of results")
):
    """Search for entities similar to the query text using vector embeddings."""
    try:
        results = vector_search.search_entities(
            query_text=query_text,
            entity_types=entity_types,
            limit=limit
        )
        
        return {"query": query_text, "results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/contextual-search")
@log_exceptions
async def contextual_search_endpoint(
    query_text: str = Body(..., description="Main query text"),
    context_text: Optional[str] = Body(None, description="Additional context text"),
    context_entity_id: Optional[str] = Body(None, description="ID of a related entity"),
    entity_types: List[str] = Body([], description="Filter by entity types"),
    limit: int = Body(10, description="Maximum number of results")
):
    """Perform search with additional context to improve relevance."""
    try:
        results = vector_search.contextual_search(
            query_text=query_text,
            context_text=context_text,
            context_entity_id=context_entity_id,
            entity_types=entity_types,
            limit=limit
        )
        
        return {"query": query_text, "results": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Error in contextual search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin Endpoints
@router.post("/update-embeddings")
@log_exceptions
async def update_embeddings(
    entity_ids: Optional[List[str]] = Body(None, description="List of entity IDs to update"),
    entity_type: Optional[str] = Body(None, description="Filter by entity type"),
    batch_size: int = Body(50, description="Number of entities to process in each batch")
):
    """Update vector embeddings for a batch of entities."""
    try:
        result = kb.update_embeddings_batch(
            entity_ids=entity_ids,
            entity_type=entity_type,
            batch_size=batch_size
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Error updating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
@log_exceptions
async def get_kb_stats(entity_type: Optional[str] = Query(None, description="Filter by entity type")):
    """Get statistics about entities in the knowledge base."""
    try:
        stats = kb.entity_stats(entity_type)
        return stats
    
    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-import")
@log_exceptions
async def bulk_import_entities(entities_data: List[Dict] = Body(..., description="List of entity data dictionaries")):
    """Import multiple entities in bulk."""
    try:
        result = kb.bulk_import_entities(entities_data)
        return result
    
    except Exception as e:
        logger.error(f"Error in bulk import: {e}")
        raise HTTPException(status_code=500, detail=str(e))