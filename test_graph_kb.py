#!/usr/bin/env python3
"""
Test script for the Graph Knowledge Base functionality
This script tests various aspects of the Graph KB after database setup
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_graph_kb")

# Load environment variables
load_dotenv()

# Import the Supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.database.supabase_client import supabase
from app.models.vector_search import VectorSearch

def test_connection():
    """Test the connection to Supabase"""
    try:
        # Try a simple query
        response = supabase.table("entities").select("*").limit(1).execute()
        logger.info(f"Connection successful, response: {response}")
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

def test_execute_sql():
    """Test the execute_sql function"""
    try:
        sql = "SELECT schema_name FROM information_schema.schemata"
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        logger.info(f"Execute SQL successful, schemas: {response.data}")
        return True
    except Exception as e:
        logger.error(f"Execute SQL test failed: {e}")
        return False
    
def get_or_create_test_user():
    """Get or create a test user for entity creation"""
    try:
        test_phone = "+1234567890"  # Test phone number
        
        # Try to get existing user
        response = supabase.table('users').select('*').eq('phone_number', test_phone).execute()
        
        if response.data and len(response.data) > 0:
            # User exists
            test_user = response.data[0]
            logger.info(f"Using existing test user with ID {test_user['id']}")
            return test_user['id']
        
        # User doesn't exist, create one
        response = supabase.table('users').insert({
            "phone_number": test_phone,
            "role": "test",
            "metadata": {}
        }).execute()
        
        if response.data and len(response.data) > 0:
            test_user = response.data[0]
            logger.info(f"Created test user with ID {test_user['id']}")
            return test_user['id']
        
        return None
    except Exception as e:
        logger.error(f"Error creating test user: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def add_test_entities():
    """Add some test entities to the Graph KB"""
    try:
        # Get or create a test user first
        test_user_id = get_or_create_test_user()
        if not test_user_id:
            logger.error("Failed to get or create test user")
            return []
            
        # Initialize embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Test entities
        entities = [
            {
                "entity_type": "business",
                "entity_name": "ABC Plumbing",
                "attributes": json.dumps({
                    "description": "Professional plumbing services",
                    "phone": "+27731234567",
                    "address": "123 Main St, Johannesburg"
                }),
                "tags": ["plumbing", "services", "repair"]
            },
            # ... rest of the entities remain the same
        ]
        
        added_ids = []
        for entity in entities:
            # Generate embedding
            text_for_embedding = f"{entity['entity_name']} {json.loads(entity['attributes'])['description']}"
            embedding = VectorSearch.get_embedding_for_text(text_for_embedding)
            
            # Check if the embedding is already a list
            embedding_list = embedding if isinstance(embedding, list) else embedding.tolist()
            
            # Call the add_entity function with valid user ID
            response = supabase.rpc("add_entity", {
                "p_entity_type": entity["entity_type"],
                "p_entity_name": entity["entity_name"],
                "p_attributes": entity["attributes"],
                "p_created_by": test_user_id,  # Use our new test user ID here instead of hardcoded UUID
                "p_tags": entity["tags"],
                "p_vector_embedding": embedding_list
            }).execute()
            
            entity_id = response.data
            added_ids.append(entity_id)
            logger.info(f"Added entity {entity['entity_name']} with ID {entity_id}")
            
        # The rest of the function remains unchanged
        # Create a relationship between two entities
        if len(added_ids) >= 2:
            response = supabase.rpc("add_relationship", {
                "p_from_entity_id": added_ids[0],
                "p_to_entity_id": added_ids[1],
                "p_relationship_type": "sells",
                "p_attributes": json.dumps({"notes": "Regular supplier"}),
                "p_weight": 0.8
            }).execute()
            
            logger.info(f"Added relationship with ID {response.data}")
            
        return added_ids
    except Exception as e:
        logger.error(f"Adding test entities failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def test_search_entities(entity_ids):
    """Test searching for entities"""
    try:
        if not entity_ids:
            logger.warning("No entity IDs provided for search test")
            return False
            
        # Test text-based search
        response = supabase.rpc("search_entities", {
            "p_search_text": "web",
            "p_limit": 5
        }).execute()
        
        logger.info(f"Text search results: {response.data}")
        
        # Test vector search (using sentence transformer)
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = VectorSearch.get_embedding_for_text("professional services")
        # Make sure it's a list
        if not isinstance(query_embedding, list):
            query_embedding = query_embedding.tolist()
        
        response = supabase.rpc("search_entities", {
            "p_search_text": None,
            "p_vector_embedding": query_embedding,
            "p_limit": 5
        }).execute()
        
        logger.info(f"Vector search results: {response.data}")
        
        # Test getting entity with relationships
        entity_id = entity_ids[0]
        response = supabase.rpc("get_entity_with_relationships", {
            "p_entity_id": entity_id,
            "p_max_depth": 1,
            "p_max_relationships_per_entity": 5
        }).execute()
        
        logger.info(f"Entity with relationships: {response.data}")
        
        return True
    except Exception as e:
        logger.error(f"Search test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_cleanup(entity_ids):
    """Clean up test entities"""
    try:
        for entity_id in entity_ids:
            response = supabase.table("entities").delete().eq("id", entity_id).execute()
            logger.info(f"Deleted entity {entity_id}")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence"""
    logger.info("Starting Graph KB tests...")
    
    # Test connection
    logger.info("1. Testing connection...")
    if not test_connection():
        logger.error("Connection test failed, aborting further tests")
        return False
        
    # Test execute_sql function
    logger.info("2. Testing execute_sql function...")
    if not test_execute_sql():
        logger.error("execute_sql test failed, aborting further tests")
        return False
        
    # Add test entities
    logger.info("3. Adding test entities...")
    entity_ids = add_test_entities()
    if not entity_ids:
        logger.error("Failed to add test entities, aborting further tests")
        return False
        
    # Test search
    logger.info("4. Testing search functionality...")
    if not test_search_entities(entity_ids):
        logger.error("Search test failed")
        # Continue to cleanup
    
    # Cleanup
    logger.info("5. Cleaning up test data...")
    test_cleanup(entity_ids)
    
    logger.info("All tests completed!")
    return True

if __name__ == "__main__":
    run_all_tests()