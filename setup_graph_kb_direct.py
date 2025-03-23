#!/usr/bin/env python3
"""
Setup script for the Graph Knowledge Base (Direct Version)

This script initializes the graph-based knowledge base database structure
by directly using the Supabase REST API rather than relying on the execute_sql RPC.

Usage:
  python setup_graph_kb_direct.py --check-dependencies
  python setup_graph_kb_direct.py --check-connection
  python setup_graph_kb_direct.py --add-sample-data
  python setup_graph_kb_direct.py --update-embeddings
"""

import argparse
import json
import logging
import sys
import importlib.util
import uuid
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# List of required dependencies for the knowledge base
REQUIRED_DEPENDENCIES = [
    ("numpy", "numpy"),
    ("supabase", "supabase-py"),
    ("langchain_community.embeddings", "langchain-community"),
    ("sentence_transformers", "sentence-transformers"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface-hub")
]

def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Tuple containing:
            - Boolean indicating if all dependencies are installed
            - List of missing dependencies
    """
    missing_deps = []
    
    logger.info("Checking required dependencies for graph knowledge base...")
    for module_name, package_name in REQUIRED_DEPENDENCIES:
        try:
            # Check if module can be imported
            importlib.util.find_spec(module_name.split('.')[0])
            logger.info(f"✓ {package_name} is installed")
        except ImportError:
            missing_deps.append(package_name)
            logger.error(f"✗ {package_name} is not installed")
    
    return len(missing_deps) == 0, missing_deps

def check_connection():
    """Check if we can connect to the Supabase database"""
    try:
        from app.database.supabase_client import supabase
        
        # Try a simple query to test connection
        response = supabase.table("users").select("*").limit(1).execute()
        logger.info(f"✓ Successfully connected to Supabase database")
        logger.info(f"✓ Found {len(response.data)} user records")
        
        # Check if entity tables exist
        try:
            response = supabase.table("entities").select("count").limit(1).execute()
            logger.info(f"✓ Entities table exists with {response.data} records")
        except Exception as e:
            logger.warning(f"Entities table not found: {e}")
            logger.warning("You need to create the required tables in Supabase")
            logger.warning("Run the setup_supabase.sql script in the Supabase SQL Editor")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {e}")
        logger.error("Please check your .env file for SUPABASE_URL and SUPABASE_KEY")
        return False

# Sample data for seeding the database
SAMPLE_ENTITIES = [
    {
        "entity_type": "provider",
        "entity_name": "Community Health Clinic",
        "attributes": {
            "location": "123 Main St, Harare",
            "phone": "+263 123 456 7890",
            "services": ["primary care", "vaccinations", "health education"],
            "hours": "Mon-Fri 8am-5pm",
            "description": "A community health clinic providing affordable healthcare services."
        },
        "tags": ["healthcare", "clinic", "community"]
    },
    {
        "entity_type": "provider",
        "entity_name": "Women's Support Center",
        "attributes": {
            "location": "456 Oak Ave, Harare",
            "phone": "+263 123 555 7890",
            "services": ["counseling", "support groups", "resources"],
            "hours": "Mon-Sat 9am-7pm",
            "description": "Support center dedicated to empowering women through various services and resources."
        },
        "tags": ["women", "support", "resources"]
    },
    {
        "entity_type": "service",
        "entity_name": "Mental Health Counseling",
        "attributes": {
            "cost": "Sliding scale",
            "eligibility": "Open to all",
            "description": "Professional counseling services for mental health concerns."
        },
        "tags": ["mental health", "counseling", "wellness"]
    },
    {
        "entity_type": "product",
        "entity_name": "Affordable Medications",
        "attributes": {
            "price_range": "$1-$20",
            "availability": "In stock",
            "description": "Low-cost generic medications for common health conditions."
        },
        "tags": ["medication", "affordable", "health"]
    },
    {
        "entity_type": "event",
        "entity_name": "Community Health Fair",
        "attributes": {
            "date": "2025-06-15",
            "time": "10:00-16:00",
            "location": "Central Park, Harare",
            "description": "Annual health fair offering free screenings, information, and resources."
        },
        "tags": ["health fair", "community", "free"]
    },
    {
        "entity_type": "event",
        "entity_name": "Women's Empowerment Workshop",
        "attributes": {
            "date": "2025-07-12",
            "time": "13:00-17:00",
            "location": "Community Center, 789 Pine St, Harare",
            "description": "Workshop focused on skills development and empowerment for women."
        },
        "tags": ["workshop", "women", "empowerment"]
    },
    {
        "entity_type": "provider",
        "entity_name": "Youth Resource Center",
        "attributes": {
            "location": "321 Elm St, Harare",
            "phone": "+263 123 777 8888",
            "services": ["education", "job training", "recreation"],
            "hours": "Mon-Fri 10am-8pm, Sat 12pm-6pm",
            "description": "Resource center dedicated to supporting youth through various programs."
        },
        "tags": ["youth", "education", "resources"]
    },
    {
        "entity_type": "service",
        "entity_name": "Job Skills Training",
        "attributes": {
            "duration": "8 weeks",
            "cost": "Free",
            "eligibility": "Ages 16-25",
            "description": "Comprehensive job skills training program for young adults."
        },
        "tags": ["training", "job skills", "youth"]
    }
]

# Sample relationships between entities
SAMPLE_RELATIONSHIPS = [
    {
        "from_type": "provider",
        "from_name": "Community Health Clinic",
        "to_type": "service",
        "to_name": "Mental Health Counseling",
        "relationship_type": "offers",
        "attributes": {
            "days_available": ["Monday", "Wednesday", "Friday"],
            "notes": "By appointment only"
        },
        "weight": 0.9
    },
    {
        "from_type": "provider",
        "from_name": "Community Health Clinic",
        "to_type": "product",
        "to_name": "Affordable Medications",
        "relationship_type": "provides",
        "attributes": {
            "prescription_required": True
        },
        "weight": 0.8
    },
    {
        "from_type": "provider",
        "from_name": "Women's Support Center",
        "to_type": "service",
        "to_name": "Mental Health Counseling",
        "relationship_type": "offers",
        "attributes": {
            "days_available": ["Tuesday", "Thursday", "Saturday"],
            "women_only": True
        },
        "weight": 0.9
    },
    {
        "from_type": "provider",
        "from_name": "Women's Support Center",
        "to_type": "event",
        "to_name": "Women's Empowerment Workshop",
        "relationship_type": "organizes",
        "attributes": {
            "registration_required": True
        },
        "weight": 1.0
    },
    {
        "from_type": "provider",
        "from_name": "Youth Resource Center",
        "to_type": "service",
        "to_name": "Job Skills Training",
        "relationship_type": "offers",
        "attributes": {
            "next_session_start": "2025-08-01"
        },
        "weight": 0.9
    },
    {
        "from_type": "provider",
        "from_name": "Youth Resource Center",
        "to_type": "event",
        "to_name": "Community Health Fair",
        "relationship_type": "participates_in",
        "attributes": {
            "booth_number": "C12"
        },
        "weight": 0.7
    }
]

def add_sample_data():
    """Add sample entities and relationships to the knowledge base directly using Supabase API."""
    try:
        from app.database.supabase_client import supabase
        from app.models.vector_search import VectorSearch
        
        # Import HuggingFace embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Dictionary to store created entities
        entity_map = {}
        
        # Add sample entities
        logger.info("Adding sample entities...")
        for entity_data in SAMPLE_ENTITIES:
            try:
                # Generate embedding if embedding model is available
                vector_embedding = None
                try:
                    # Create text representation for embedding
                    text_to_embed = f"{entity_data['entity_name']}. "
                    
                    # Add attributes to text representation
                    for key, value in entity_data["attributes"].items():
                        if isinstance(value, (str, int, float, bool)):
                            text_to_embed += f"{key}: {value}. "
                    
                    # Generate embedding
                    vector_embedding = embedding_model.embed_query(text_to_embed)
                    logger.info(f"Generated embedding for {entity_data['entity_name']}")
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                
                # Create entity data for insertion
                insert_data = {
                    "entity_type": entity_data["entity_type"],
                    "entity_name": entity_data["entity_name"],
                    "attributes": entity_data["attributes"],
                    "tags": entity_data["tags"],
                    "vector_embedding": vector_embedding
                }
                
                # Insert entity using Supabase
                response = supabase.table("entities").insert(insert_data).execute()
                
                if response.data and len(response.data) > 0:
                    entity_id = response.data[0]["id"]
                    # Store in map for relationship creation
                    key = f"{entity_data['entity_type']}:{entity_data['entity_name']}"
                    entity_map[key] = entity_id
                    logger.info(f"Added entity: {entity_data['entity_name']} (ID: {entity_id})")
                else:
                    logger.error(f"Failed to add entity: {entity_data['entity_name']}")
            except Exception as e:
                logger.error(f"Error adding entity {entity_data['entity_name']}: {e}")
        
        # Add sample relationships
        logger.info("Adding sample relationships...")
        for rel_data in SAMPLE_RELATIONSHIPS:
            try:
                # Get entity IDs
                from_key = f"{rel_data['from_type']}:{rel_data['from_name']}"
                to_key = f"{rel_data['to_type']}:{rel_data['to_name']}"
                
                from_id = entity_map.get(from_key)
                to_id = entity_map.get(to_key)
                
                if not from_id or not to_id:
                    logger.error(f"Cannot create relationship: missing entity {from_key} or {to_key}")
                    continue
                
                # Create relationship data for insertion
                insert_data = {
                    "from_entity_id": from_id,
                    "to_entity_id": to_id,
                    "relationship_type": rel_data["relationship_type"],
                    "attributes": rel_data.get("attributes", {}),
                    "weight": rel_data.get("weight", 1.0)
                }
                
                # Insert relationship using Supabase
                response = supabase.table("entity_relationships").insert(insert_data).execute()
                
                if response.data and len(response.data) > 0:
                    rel_id = response.data[0]["id"]
                    logger.info(f"Added relationship: {rel_data['from_name']} --[{rel_data['relationship_type']}]--> {rel_data['to_name']} (ID: {rel_id})")
                else:
                    logger.error(f"Failed to add relationship between {rel_data['from_name']} and {rel_data['to_name']}")
            except Exception as e:
                logger.error(f"Error adding relationship: {e}")
        
        logger.info("Sample data added successfully.")
        return True
    except Exception as e:
        logger.error(f"Error adding sample data: {e}")
        return False

def update_embeddings():
    """Update vector embeddings for all entities directly using Supabase API."""
    try:
        from app.database.supabase_client import supabase
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Get all entities without embeddings
        response = supabase.table("entities").select("id,entity_name,attributes").execute()
        entities = response.data or []
        
        logger.info(f"Found {len(entities)} entities to process")
        
        total_updated = 0
        total_failed = 0
        batch_size = 10
        
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
                    update_response = supabase.table("entities").update(
                        {"vector_embedding": embedding, "updated_at": "now()"}
                    ).eq("id", entity_id).execute()
                    
                    if update_response.data and len(update_response.data) > 0:
                        total_updated += 1
                        logger.info(f"Updated embedding for {entity_name}")
                    else:
                        total_failed += 1
                        logger.error(f"Failed to update embedding for entity {entity_id}")
                        
                except Exception as e:
                    total_failed += 1
                    logger.error(f"Error updating embedding for entity {entity.get('id')}: {e}")
            
            logger.info(f"Processed {i + len(batch)}/{len(entities)} entities")
        
        result = {
            "total": len(entities),
            "updated": total_updated,
            "failed": total_failed
        }
        
        logger.info(f"Embedding update results: {json.dumps(result)}")
        return result
    except Exception as e:
        logger.error(f"Error updating embeddings batch: {e}")
        return {"error": str(e), "updated": 0, "failed": 0}

def main():
    parser = argparse.ArgumentParser(description="Set up the graph-based knowledge base (direct version)")
    parser.add_argument("--check-dependencies", action="store_true", help="Check if all required dependencies are installed")
    parser.add_argument("--check-connection", action="store_true", help="Check database connection and tables")
    parser.add_argument("--add-sample-data", action="store_true", help="Add sample data to the knowledge base")
    parser.add_argument("--update-embeddings", action="store_true", help="Update vector embeddings for all entities")
    
    args = parser.parse_args()
    
    # Handle dependency check first
    if args.check_dependencies:
        all_deps_installed, missing_deps = check_dependencies()
        if not all_deps_installed:
            logger.error("Missing dependencies detected.")
            logger.error("Please install the missing packages with pip:")
            logger.error(f"pip install {' '.join(missing_deps)}")
            logger.error("Or install all dependencies with:")
            logger.error("pip install -r requirements.txt")
            return 1
        else:
            logger.info("All dependencies are installed correctly!")
            return 0
    
    # Load environment variables
    load_dotenv()
    
    # Quick dependency check if we're performing other operations
    all_deps_installed, missing_deps = check_dependencies()
    if not all_deps_installed:
        logger.error("Missing dependencies detected. Run with --check-dependencies for details.")
        logger.error("Please install all dependencies with: pip install -r requirements.txt")
        return 1
    
    if args.check_connection:
        if not check_connection():
            logger.error("Database connection check failed.")
            return 1
        else:
            logger.info("Database connection check passed.")
    
    if args.add_sample_data:
        if not add_sample_data():
            logger.error("Failed to add sample data.")
            return 1
        else:
            logger.info("Sample data added successfully.")
    
    if args.update_embeddings:
        if not update_embeddings():
            logger.error("Failed to update embeddings.")
            return 1
        else:
            logger.info("Embeddings updated successfully.")
    
    if not (args.check_dependencies or args.check_connection or args.add_sample_data or args.update_embeddings):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())