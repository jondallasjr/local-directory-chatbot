"""
Graph-based Knowledge Base with Vector Search

This module implements a graph-based knowledge base structure with vector embedding search
capabilities for the Directory Chatbot. It provides functionality to:

1. Create and manage entities in a graph structure
2. Store and retrieve vector embeddings for semantic search
3. Traverse the graph with variable depth/width
4. Execute complex queries combining vector similarity with graph relationships
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set, Union

import numpy as np
from supabase import Client
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.database.supabase_client import supabase, SupabaseDB
from app.utils.logging_utils import log_exceptions

# Configure logging
logger = logging.getLogger(__name__)

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embedding model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing embedding model: {e}")
    embedding_model = None


class GraphKnowledgeBase:
    """
    Graph-based Knowledge Base with vector search capabilities.
    This class provides methods for managing a graph structure of entities
    and performing semantic search using vector embeddings.
    """
    
    def __init__(self, supabase_client: Client = supabase):
        """
        Initialize the knowledge base with a Supabase client.
        
        Args:
            supabase_client: Supabase client instance
        """
        self.supabase = supabase_client
        
    @log_exceptions
    def create_entity_tables(self) -> bool:
        """
        Create the necessary tables for the graph knowledge base.
        This should be run once during initial setup.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create entities table if not exists
            entities_table_sql = """
            CREATE TABLE IF NOT EXISTS "entities" (
                "id" uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                "entity_type" text NOT NULL,
                "entity_name" text NOT NULL,
                "attributes" jsonb DEFAULT '{}'::jsonb,
                "vector_embedding" vector(384),
                "created_by" uuid,
                "verification_status" text DEFAULT 'unverified',
                "metadata" jsonb DEFAULT '{}'::jsonb,
                "tags" text[] DEFAULT '{}'::text[],
                "created_at" timestamptz DEFAULT now(),
                "updated_at" timestamptz DEFAULT now()
            );
            """
            
            # Create entity relationships table if not exists
            relationships_table_sql = """
            CREATE TABLE IF NOT EXISTS "entity_relationships" (
                "id" uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                "from_entity_id" uuid NOT NULL REFERENCES "entities" ("id") ON DELETE CASCADE,
                "to_entity_id" uuid NOT NULL REFERENCES "entities" ("id") ON DELETE CASCADE,
                "relationship_type" text NOT NULL,
                "attributes" jsonb DEFAULT '{}'::jsonb,
                "weight" real DEFAULT 1.0,
                "created_at" timestamptz DEFAULT now(),
                "updated_at" timestamptz DEFAULT now(),
                UNIQUE ("from_entity_id", "to_entity_id", "relationship_type")
            );
            """
            
            # Execute SQL to create tables
            self.supabase.rpc("execute_sql", {"sql": entities_table_sql}).execute()
            self.supabase.rpc("execute_sql", {"sql": relationships_table_sql}).execute()
            
            # Create vector index on entities table
            vector_index_sql = """
            CREATE INDEX IF NOT EXISTS "entities_vector_idx" 
            ON "entities" 
            USING ivfflat (vector_embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            self.supabase.rpc("execute_sql", {"sql": vector_index_sql}).execute()
            
            # Create indices for faster queries
            indices_sql = [
                'CREATE INDEX IF NOT EXISTS "entities_entity_type_idx" ON "entities" ("entity_type");',
                'CREATE INDEX IF NOT EXISTS "entities_entity_name_idx" ON "entities" ("entity_name" text_pattern_ops);',
                'CREATE INDEX IF NOT EXISTS "entity_relationships_from_idx" ON "entity_relationships" ("from_entity_id");',
                'CREATE INDEX IF NOT EXISTS "entity_relationships_to_idx" ON "entity_relationships" ("to_entity_id");',
                'CREATE INDEX IF NOT EXISTS "entity_relationships_type_idx" ON "entity_relationships" ("relationship_type");'
            ]
            
            for sql in indices_sql:
                self.supabase.rpc("execute_sql", {"sql": sql}).execute()
            
            logger.info("Knowledge base tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating knowledge base tables: {e}")
            return False
    
    @log_exceptions
    def add_entity(self, 
                  entity_type: str, 
                  entity_name: str, 
                  attributes: Dict = None, 
                  created_by: str = None,
                  tags: List[str] = None,
                  generate_embedding: bool = True) -> Optional[str]:
        """
        Add a new entity to the knowledge base.
        
        Args:
            entity_type: Type of entity (e.g., 'person', 'organization', 'event')
            entity_name: Name of the entity
            attributes: Dictionary of entity attributes
            created_by: ID of the user who created the entity
            tags: List of tags for the entity
            generate_embedding: Whether to generate and store a vector embedding
            
        Returns:
            ID of the created entity, or None if failed
        """
        try:
            attributes = attributes or {}
            tags = tags or []
            
            entity_data = {
                "entity_type": entity_type,
                "entity_name": entity_name,
                "attributes": attributes,
                "created_by": created_by,
                "tags": tags
            }
            
            # Generate vector embedding if requested
            if generate_embedding and embedding_model:
                try:
                    # Create text representation for embedding
                    text_to_embed = f"{entity_name}. "
                    
                    # Add attributes to text representation
                    for key, value in attributes.items():
                        if isinstance(value, (str, int, float, bool)):
                            text_to_embed += f"{key}: {value}. "
                    
                    # Generate embedding
                    embedding = embedding_model.embed_query(text_to_embed)
                    
                    # Add to entity data
                    entity_data["vector_embedding"] = embedding
                except Exception as e:
                    logger.error(f"Error generating embedding for entity {entity_name}: {e}")
            
            # Insert entity into database
            response = self.supabase.table("entities").insert(entity_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Entity created: {entity_name} ({entity_type})")
                return response.data[0]["id"]
            else:
                logger.error(f"Failed to create entity: {entity_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error adding entity: {e}")
            return None
    
    @log_exceptions
    def update_entity(self, 
                     entity_id: str, 
                     entity_type: Optional[str] = None,
                     entity_name: Optional[str] = None,
                     attributes: Optional[Dict] = None,
                     tags: Optional[List[str]] = None,
                     update_embedding: bool = False) -> bool:
        """
        Update an existing entity in the knowledge base.
        
        Args:
            entity_id: ID of the entity to update
            entity_type: New entity type (if changing)
            entity_name: New entity name (if changing)
            attributes: New attributes (will be merged with existing)
            tags: New tags (will replace existing)
            update_embedding: Whether to update the vector embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current entity data
            entity = self.get_entity(entity_id)
            if not entity:
                logger.error(f"Entity not found: {entity_id}")
                return False
            
            # Prepare update data
            update_data = {}
            
            if entity_type:
                update_data["entity_type"] = entity_type
            
            if entity_name:
                update_data["entity_name"] = entity_name
            
            if attributes:
                # Merge with existing attributes
                existing_attributes = entity.get("attributes", {})
                merged_attributes = {**existing_attributes, **attributes}
                update_data["attributes"] = merged_attributes
            
            if tags is not None:  # Allow empty list to clear tags
                update_data["tags"] = tags
            
            # Update timestamp
            update_data["updated_at"] = "now()"
            
            # Update embedding if requested
            if update_embedding and embedding_model:
                try:
                    name = entity_name or entity.get("entity_name", "")
                    attrs = attributes or entity.get("attributes", {})
                    
                    # Create text representation for embedding
                    text_to_embed = f"{name}. "
                    
                    # Add attributes to text representation
                    for key, value in attrs.items():
                        if isinstance(value, (str, int, float, bool)):
                            text_to_embed += f"{key}: {value}. "
                    
                    # Generate embedding
                    embedding = embedding_model.embed_query(text_to_embed)
                    
                    # Add to update data
                    update_data["vector_embedding"] = embedding
                except Exception as e:
                    logger.error(f"Error updating embedding for entity {entity_id}: {e}")
            
            # Update entity in database
            response = self.supabase.table("entities").update(update_data).eq("id", entity_id).execute()
            
            success = response.data and len(response.data) > 0
            if success:
                logger.info(f"Entity updated: {entity_id}")
            else:
                logger.error(f"Failed to update entity: {entity_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating entity: {e}")
            return False
    
    @log_exceptions
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the knowledge base.
        Note: This will also delete all relationships involving this entity.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete entity from database
            response = self.supabase.table("entities").delete().eq("id", entity_id).execute()
            
            success = response.data and len(response.data) > 0
            if success:
                logger.info(f"Entity deleted: {entity_id}")
            else:
                logger.error(f"Failed to delete entity: {entity_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting entity: {e}")
            return False
    
    @log_exceptions
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity data or None if not found
        """
        try:
            response = self.supabase.table("entities").select("*").eq("id", entity_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            else:
                logger.warning(f"Entity not found: {entity_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            return None
    
    @log_exceptions
    def add_relationship(self, 
                        from_entity_id: str, 
                        to_entity_id: str, 
                        relationship_type: str,
                        attributes: Dict = None,
                        weight: float = 1.0) -> Optional[str]:
        """
        Add a relationship between two entities.
        
        Args:
            from_entity_id: ID of the source entity
            to_entity_id: ID of the target entity
            relationship_type: Type of relationship
            attributes: Dictionary of relationship attributes
            weight: Weight/strength of the relationship (0.0 to 1.0)
            
        Returns:
            ID of the created relationship, or None if failed
        """
        try:
            attributes = attributes or {}
            
            # Check if both entities exist
            from_entity = self.get_entity(from_entity_id)
            to_entity = self.get_entity(to_entity_id)
            
            if not from_entity:
                logger.error(f"Source entity not found: {from_entity_id}")
                return None
            
            if not to_entity:
                logger.error(f"Target entity not found: {to_entity_id}")
                return None
            
            # Create relationship data
            relationship_data = {
                "from_entity_id": from_entity_id,
                "to_entity_id": to_entity_id,
                "relationship_type": relationship_type,
                "attributes": attributes,
                "weight": max(0.0, min(1.0, weight))  # Clamp weight between 0 and 1
            }
            
            # Insert relationship into database
            response = self.supabase.table("entity_relationships").insert(relationship_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Relationship created: {from_entity_id} --[{relationship_type}]--> {to_entity_id}")
                return response.data[0]["id"]
            else:
                logger.error(f"Failed to create relationship between {from_entity_id} and {to_entity_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
            return None
    
    @log_exceptions
    def delete_relationship(self, 
                           from_entity_id: str, 
                           to_entity_id: str, 
                           relationship_type: str = None) -> bool:
        """
        Delete a relationship between two entities.
        
        Args:
            from_entity_id: ID of the source entity
            to_entity_id: ID of the target entity
            relationship_type: Type of relationship (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build query
            query = self.supabase.table("entity_relationships").delete().eq("from_entity_id", from_entity_id).eq("to_entity_id", to_entity_id)
            
            # Add relationship type filter if provided
            if relationship_type:
                query = query.eq("relationship_type", relationship_type)
            
            # Execute query
            response = query.execute()
            
            success = response.data and len(response.data) > 0
            if success:
                if relationship_type:
                    logger.info(f"Relationship deleted: {from_entity_id} --[{relationship_type}]--> {to_entity_id}")
                else:
                    logger.info(f"All relationships deleted between {from_entity_id} and {to_entity_id}")
            else:
                logger.error(f"Failed to delete relationship(s) between {from_entity_id} and {to_entity_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting relationship: {e}")
            return False
    
    @log_exceptions
    def get_relationships(self, 
                         entity_id: str, 
                         relationship_type: Optional[str] = None,
                         direction: str = "outgoing") -> List[Dict]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            relationship_type: Filter by relationship type (optional)
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relationship data
        """
        try:
            results = []
            
            # Query outgoing relationships
            if direction in ["outgoing", "both"]:
                query = self.supabase.table("entity_relationships").select("*").eq("from_entity_id", entity_id)
                
                if relationship_type:
                    query = query.eq("relationship_type", relationship_type)
                
                response = query.execute()
                if response.data:
                    for rel in response.data:
                        rel["direction"] = "outgoing"
                        results.append(rel)
            
            # Query incoming relationships
            if direction in ["incoming", "both"]:
                query = self.supabase.table("entity_relationships").select("*").eq("to_entity_id", entity_id)
                
                if relationship_type:
                    query = query.eq("relationship_type", relationship_type)
                
                response = query.execute()
                if response.data:
                    for rel in response.data:
                        rel["direction"] = "incoming"
                        results.append(rel)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
    
    @log_exceptions
    def search_entities(self, 
                       search_text: str = None,
                       entity_type: str = None,
                       tags: List[str] = None, 
                       limit: int = 10,
                       use_vector_search: bool = True) -> List[Dict]:
        """
        Search for entities based on text, type, and tags.
        
        Args:
            search_text: Text to search for
            entity_type: Filter by entity type
            tags: Filter by tags
            limit: Maximum number of results
            use_vector_search: Whether to use vector search (if available)
            
        Returns:
            List of matching entities
        """
        try:
            # Use vector search if requested and available
            if search_text and use_vector_search and embedding_model:
                try:
                    # Generate embedding for search query
                    query_embedding = embedding_model.embed_query(search_text)
                    
                    # Build the vector search query
                    sql = """
                    SELECT *, 1 - (vector_embedding <=> $1) AS similarity
                    FROM entities
                    WHERE 1=1
                    """
                    
                    # Add entity type filter if provided
                    params = [query_embedding]
                    
                    if entity_type:
                        sql += " AND entity_type = $2"
                        params.append(entity_type)
                    
                    # Add tags filter if provided
                    if tags and len(tags) > 0:
                        tag_idx = len(params) + 1
                        sql += f" AND tags && ${tag_idx}::text[]"
                        params.append(tags)
                    
                    # Add order and limit
                    sql += " ORDER BY similarity DESC LIMIT $" + str(len(params) + 1)
                    params.append(limit)
                    
                    # Execute the query
                    response = self.supabase.rpc("execute_sql", {"sql": sql, "params": params}).execute()
                    
                    if response.data:
                        return response.data
                    
                except Exception as e:
                    logger.error(f"Error in vector search: {e}")
                    # Fall back to regular search
            
            # Regular database search
            query = self.supabase.table("entities").select("*")
            
            # Add filters
            if entity_type:
                query = query.eq("entity_type", entity_type)
            
            if search_text:
                query = query.ilike("entity_name", f"%{search_text}%")
            
            if tags and len(tags) > 0:
                # Filter for entities with any of the provided tags
                for tag in tags:
                    query = query.contains('tags', [tag])
            
            # Execute query
            response = query.limit(limit).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
    
    @log_exceptions
    def traversal_search(self, 
                        start_entity_id: str,
                        max_depth: int = 2,
                        max_breadth: int = 5,
                        relationship_types: List[str] = None) -> Dict:
        """
        Perform a breadth-first graph traversal from a starting entity.
        
        Args:
            start_entity_id: ID of the starting entity
            max_depth: Maximum traversal depth
            max_breadth: Maximum number of relationships to follow per entity
            relationship_types: Filter by relationship types
            
        Returns:
            Dictionary with traversal results
        """
        try:
            # Check if start entity exists
            start_entity = self.get_entity(start_entity_id)
            if not start_entity:
                logger.error(f"Start entity not found: {start_entity_id}")
                return {"error": "Start entity not found", "nodes": [], "edges": []}
            
            # Initialize traversal
            visited = set([start_entity_id])
            nodes = {start_entity_id: {"entity": start_entity, "depth": 0}}
            edges = []
            queue = [(start_entity_id, 0)]  # (entity_id, depth)
            
            # Breadth-first traversal
            while queue and max_depth > 0:
                current_id, current_depth = queue.pop(0)
                
                # Stop if max depth reached
                if current_depth >= max_depth:
                    continue
                
                # Get relationships for current entity
                relationships = self.get_relationships(current_id, direction="both")
                
                # Filter by relationship types if specified
                if relationship_types:
                    relationships = [r for r in relationships if r["relationship_type"] in relationship_types]
                
                # Sort by weight (descending) and limit by max_breadth
                relationships.sort(key=lambda x: x.get("weight", 0.0), reverse=True)
                relationships = relationships[:max_breadth]
                
                # Process each relationship
                for rel in relationships:
                    # Determine the connected entity
                    if rel["direction"] == "outgoing":
                        connected_id = rel["to_entity_id"]
                        edge_direction = "outgoing"
                    else:
                        connected_id = rel["from_entity_id"]
                        edge_direction = "incoming"
                    
                    # Skip already visited entities
                    if connected_id in visited:
                        # Still add the edge if not already present
                        edge_key = f"{current_id}:{connected_id}:{rel['relationship_type']}"
                        edge_key_rev = f"{connected_id}:{current_id}:{rel['relationship_type']}"
                        
                        edge_exists = any(e.get("key") in [edge_key, edge_key_rev] for e in edges)
                        
                        if not edge_exists:
                            edges.append({
                                "key": edge_key,
                                "from_id": current_id if edge_direction == "outgoing" else connected_id,
                                "to_id": connected_id if edge_direction == "outgoing" else current_id,
                                "type": rel["relationship_type"],
                                "attributes": rel.get("attributes", {}),
                                "weight": rel.get("weight", 1.0)
                            })
                        
                        continue
                    
                    # Mark as visited
                    visited.add(connected_id)
                    
                    # Get connected entity
                    connected_entity = self.get_entity(connected_id)
                    if connected_entity:
                        nodes[connected_id] = {
                            "entity": connected_entity,
                            "depth": current_depth + 1
                        }
                        
                        # Add edge
                        edge_key = f"{current_id}:{connected_id}:{rel['relationship_type']}"
                        edges.append({
                            "key": edge_key,
                            "from_id": current_id if edge_direction == "outgoing" else connected_id,
                            "to_id": connected_id if edge_direction == "outgoing" else current_id,
                            "type": rel["relationship_type"],
                            "attributes": rel.get("attributes", {}),
                            "weight": rel.get("weight", 1.0)
                        })
                        
                        # Add to queue for further traversal
                        queue.append((connected_id, current_depth + 1))
            
            # Prepare result
            result = {
                "start_entity_id": start_entity_id,
                "max_depth": max_depth,
                "nodes": list(nodes.values()),
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in traversal search: {e}")
            return {"error": str(e), "nodes": [], "edges": []}
    
    @log_exceptions
    def hybrid_search(self, 
                     search_text: str,
                     entity_types: List[str] = None,
                     tags: List[str] = None,
                     expand_results: bool = True,
                     max_expansion_depth: int = 1,
                     limit: int = 10) -> Dict:
        """
        Perform a hybrid search combining vector similarity with graph expansion.
        
        Args:
            search_text: Text to search for
            entity_types: Filter by entity types
            tags: Filter by tags
            expand_results: Whether to expand results with related entities
            max_expansion_depth: Maximum depth for expanding results
            limit: Maximum number of direct search results
            
        Returns:
            Dictionary with search results
        """
        try:
            # Initial vector search
            initial_results = []
            
            if entity_types and len(entity_types) > 0:
                # Search separately for each entity type to ensure variety
                per_type_limit = max(3, limit // len(entity_types))
                
                for entity_type in entity_types:
                    type_results = self.search_entities(
                        search_text=search_text,
                        entity_type=entity_type,
                        tags=tags,
                        limit=per_type_limit,
                        use_vector_search=True
                    )
                    initial_results.extend(type_results)
                
                # Sort by similarity if available, otherwise by name
                if initial_results and "similarity" in initial_results[0]:
                    initial_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                initial_results = initial_results[:limit]
            else:
                # General search without entity type filtering
                initial_results = self.search_entities(
                    search_text=search_text,
                    tags=tags,
                    limit=limit,
                    use_vector_search=True
                )
            
            # Prepare results dictionary
            results = {
                "search_text": search_text,
                "direct_results": initial_results,
                "direct_result_count": len(initial_results),
                "expanded_results": [],
                "expanded_result_count": 0,
                "total_result_count": len(initial_results)
            }
            
            # Expand results if requested
            if expand_results and initial_results:
                expanded_entities = {}
                
                # Add initial results to expanded entities
                for entity in initial_results:
                    expanded_entities[entity["id"]] = {
                        "entity": entity,
                        "depth": 0,
                        "path": []
                    }
                
                # Expand each result
                for entity in initial_results:
                    # Skip expansion if we've reached the limit
                    if len(expanded_entities) >= limit * 3:
                        break
                    
                    # Perform traversal search from this entity
                    traversal = self.traversal_search(
                        start_entity_id=entity["id"],
                        max_depth=max_expansion_depth,
                        max_breadth=3
                    )
                    
                    # Add traversal nodes to expanded entities
                    for node in traversal.get("nodes", []):
                        node_entity = node["entity"]
                        node_id = node_entity["id"]
                        
                        # Skip initial entities
                        if node["depth"] == 0:
                            continue
                        
                        # Add if not already in expanded entities
                        if node_id not in expanded_entities:
                            expanded_entities[node_id] = {
                                "entity": node_entity,
                                "depth": node["depth"],
                                "source": entity["id"],
                                "source_name": entity["entity_name"]
                            }
                
                # Add expanded entities to results
                results["expanded_results"] = list(expanded_entities.values())
                results["expanded_result_count"] = len(results["expanded_results"])
                results["total_result_count"] = results["direct_result_count"] + results["expanded_result_count"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {
                "error": str(e),
                "search_text": search_text,
                "direct_results": [],
                "direct_result_count": 0,
                "expanded_results": [],
                "expanded_result_count": 0,
                "total_result_count": 0
            }

    @log_exceptions
    def update_embeddings_batch(self, 
                               entity_ids: List[str] = None,
                               entity_type: str = None,
                               batch_size: int = 50) -> Dict:
        """
        Update vector embeddings for a batch of entities.
        
        Args:
            entity_ids: List of entity IDs to update (optional)
            entity_type: Filter by entity type (optional)
            batch_size: Number of entities to process in each batch
            
        Returns:
            Dictionary with update results
        """
        if not embedding_model:
            return {"error": "Embedding model not available", "updated": 0, "failed": 0}
        
        try:
            total_updated = 0
            total_failed = 0
            
            # Build query to get entities
            query = self.supabase.table("entities").select("id,entity_name,attributes")
            
            if entity_ids:
                # Use in filter for specific entities
                query = query.in_("id", entity_ids)
            elif entity_type:
                # Filter by entity type
                query = query.eq("entity_type", entity_type)
            
            # Execute query
            response = query.execute()
            entities = response.data or []
            
            # Process entities in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                for entity in batch:
                    try:
                        entity_id = entity["id"]
                        entity_name = entity["entity_name"]
                        attributes = entity.get("attributes", {})
                        
                        # Create text representation for embedding
                        text_to_embed = f"{entity_name}. "
                        
                        # Add attributes to text representation
                        for key, value in attributes.items():
                            if isinstance(value, (str, int, float, bool)):
                                text_to_embed += f"{key}: {value}. "
                        
                        # Generate embedding
                        embedding = embedding_model.embed_query(text_to_embed)
                        
                        # Update entity with new embedding
                        update_response = self.supabase.table("entities").update(
                            {"vector_embedding": embedding, "updated_at": "now()"}
                        ).eq("id", entity_id).execute()
                        
                        if update_response.data and len(update_response.data) > 0:
                            total_updated += 1
                        else:
                            total_failed += 1
                            logger.error(f"Failed to update embedding for entity {entity_id}")
                            
                    except Exception as e:
                        total_failed += 1
                        logger.error(f"Error updating embedding for entity {entity.get('id')}: {e}")
                
                logger.info(f"Processed {i + len(batch)}/{len(entities)} entities")
            
            return {
                "total": len(entities),
                "updated": total_updated,
                "failed": total_failed
            }
            
        except Exception as e:
            logger.error(f"Error updating embeddings batch: {e}")
            return {"error": str(e), "updated": 0, "failed": 0}
    
    @log_exceptions
    def entity_stats(self, entity_type: str = None) -> Dict:
        """
        Get statistics about entities in the knowledge base.
        
        Args:
            entity_type: Filter by entity type (optional)
            
        Returns:
            Dictionary with statistics
        """
        try:
            stats = {}
            
            # Build query for counting entities
            entity_count_query = self.supabase.rpc(
                "execute_sql", 
                {"sql": "SELECT COUNT(*) FROM entities"}
            )
            
            if entity_type:
                entity_count_query = self.supabase.rpc(
                    "execute_sql", 
                    {"sql": "SELECT COUNT(*) FROM entities WHERE entity_type = $1", "params": [entity_type]}
                )
            
            # Execute query
            response = entity_count_query.execute()
            if response.data:
                stats["entity_count"] = response.data[0]["count"]
            else:
                stats["entity_count"] = 0
            
            # Get entity type distribution
            type_query = self.supabase.rpc(
                "execute_sql", 
                {"sql": "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY COUNT(*) DESC"}
            )
            response = type_query.execute()
            if response.data:
                stats["entity_types"] = response.data
            else:
                stats["entity_types"] = []
            
            # Get relationship count
            rel_count_query = self.supabase.rpc(
                "execute_sql", 
                {"sql": "SELECT COUNT(*) FROM entity_relationships"}
            )
            response = rel_count_query.execute()
            if response.data:
                stats["relationship_count"] = response.data[0]["count"]
            else:
                stats["relationship_count"] = 0
            
            # Get relationship type distribution
            rel_type_query = self.supabase.rpc(
                "execute_sql", 
                {"sql": "SELECT relationship_type, COUNT(*) FROM entity_relationships GROUP BY relationship_type ORDER BY COUNT(*) DESC"}
            )
            response = rel_type_query.execute()
            if response.data:
                stats["relationship_types"] = response.data
            else:
                stats["relationship_types"] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting entity stats: {e}")
            return {"error": str(e)}
    
    @log_exceptions
    def bulk_import_entities(self, entities_data: List[Dict]) -> Dict:
        """
        Import multiple entities in bulk.
        
        Args:
            entities_data: List of entity data dictionaries
            
        Returns:
            Dictionary with import results
        """
        try:
            results = {
                "total": len(entities_data),
                "successful": 0,
                "failed": 0,
                "entity_ids": []
            }
            
            for entity_data in entities_data:
                try:
                    entity_type = entity_data.get("entity_type")
                    entity_name = entity_data.get("entity_name")
                    
                    if not entity_type or not entity_name:
                        results["failed"] += 1
                        logger.error(f"Missing required fields for entity: {entity_data}")
                        continue
                    
                    # Create entity
                    entity_id = self.add_entity(
                        entity_type=entity_type,
                        entity_name=entity_name,
                        attributes=entity_data.get("attributes", {}),
                        created_by=entity_data.get("created_by"),
                        tags=entity_data.get("tags", []),
                        generate_embedding=True
                    )
                    
                    if entity_id:
                        results["successful"] += 1
                        results["entity_ids"].append(entity_id)
                    else:
                        results["failed"] += 1
                
                except Exception as e:
                    results["failed"] += 1
                    logger.error(f"Error importing entity {entity_data.get('entity_name')}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk import: {e}")
            return {"error": str(e), "total": len(entities_data), "successful": 0, "failed": len(entities_data)}