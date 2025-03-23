#!/usr/bin/env python3
"""
Setup script for the Graph Knowledge Base

This script initializes the graph-based knowledge base database structure,
creates necessary tables, indices, and functions, and optionally seeds
the database with example data.

Usage:
  python setup_graph_kb.py --create-tables
  python setup_graph_kb.py --add-sample-data
  python setup_graph_kb.py --update-embeddings
  python setup_graph_kb.py --check-dependencies
"""

import argparse
import json
import logging
import os
import sys
import time
import importlib.util
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
    ("huggingface_hub", "huggingface-hub"),
    ("pgvector", "pgvector")
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
    
    # Check for database connection and pgvector extension
    if not missing_deps:
        try:
            logger.info("Checking database connection and pgvector extension...")
            # The next imports will only work if the dependencies are installed
            from app.database.supabase_client import supabase
            
            # Check if we can connect to Supabase
            try:
                # Simple query to check connection
                response = supabase.table("_rpc").select().execute()
                logger.info("✓ Database connection is working")
                
                # Check for pgvector extension
                sql = "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                result = supabase.rpc("execute_sql", {"sql": sql}).execute()
                
                if result.data and len(result.data) > 0:
                    logger.info("✓ pgvector extension is installed")
                else:
                    logger.warning("✗ pgvector extension is not installed in the database")
                    logger.warning("Run the script with --create-pg-vector to install it")
            except Exception as e:
                logger.error(f"✗ Database connection error: {str(e)}")
                logger.error("Please check your .env file for SUPABASE_URL and SUPABASE_KEY")
        except Exception as e:
            # This should not happen if all dependencies are installed
            logger.error(f"Error checking database: {str(e)}")
    
    return len(missing_deps) == 0, missing_deps

# Only import app modules after dependency check to avoid import errors
try:
    from app.database.graph_kb import GraphKnowledgeBase
    from app.database.supabase_client import supabase
    from app.models.vector_search import VectorSearch
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Please run with --check-dependencies to identify missing packages")

# Logger is already configured above

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

def create_tables():
    """Create necessary tables for the graph knowledge base."""
    kb = GraphKnowledgeBase()
    if kb.create_entity_tables():
        logger.info("Knowledge base tables created successfully.")
    else:
        logger.error("Failed to create knowledge base tables.")

def create_pg_vector_extension():
    """Create the pgvector extension for vector search."""
    try:
        # Check if extension already exists
        sql = "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        result = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        if result.data and len(result.data) > 0:
            logger.info("pgvector extension already exists.")
            return True
        
        # Create the extension
        sql = "CREATE EXTENSION IF NOT EXISTS vector"
        supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        logger.info("pgvector extension created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating pgvector extension: {e}")
        return False

def add_sample_data():
    """Add sample entities and relationships to the knowledge base."""
    kb = GraphKnowledgeBase()
    
    # Dictionary to store created entities
    entity_map = {}
    
    # Add sample entities
    logger.info("Adding sample entities...")
    for entity_data in SAMPLE_ENTITIES:
        entity_id = kb.add_entity(
            entity_type=entity_data["entity_type"],
            entity_name=entity_data["entity_name"],
            attributes=entity_data["attributes"],
            tags=entity_data["tags"],
            generate_embedding=True
        )
        
        if entity_id:
            # Store in map for relationship creation
            key = f"{entity_data['entity_type']}:{entity_data['entity_name']}"
            entity_map[key] = entity_id
            logger.info(f"Added entity: {entity_data['entity_name']}")
        else:
            logger.error(f"Failed to add entity: {entity_data['entity_name']}")
    
    # Add sample relationships
    logger.info("Adding sample relationships...")
    for rel_data in SAMPLE_RELATIONSHIPS:
        # Get entity IDs
        from_key = f"{rel_data['from_type']}:{rel_data['from_name']}"
        to_key = f"{rel_data['to_type']}:{rel_data['to_name']}"
        
        from_id = entity_map.get(from_key)
        to_id = entity_map.get(to_key)
        
        if not from_id or not to_id:
            logger.error(f"Cannot create relationship: missing entity {from_key} or {to_key}")
            continue
        
        # Create relationship
        rel_id = kb.add_relationship(
            from_entity_id=from_id,
            to_entity_id=to_id,
            relationship_type=rel_data["relationship_type"],
            attributes=rel_data.get("attributes", {}),
            weight=rel_data.get("weight", 1.0)
        )
        
        if rel_id:
            logger.info(f"Added relationship: {rel_data['from_name']} --[{rel_data['relationship_type']}]--> {rel_data['to_name']}")
        else:
            logger.error(f"Failed to add relationship between {rel_data['from_name']} and {rel_data['to_name']}")

def update_embeddings():
    """Update vector embeddings for all entities."""
    kb = GraphKnowledgeBase()
    result = kb.update_embeddings_batch()
    
    logger.info(f"Embedding update results: {json.dumps(result)}")

def main():
    parser = argparse.ArgumentParser(description="Set up the graph-based knowledge base")
    parser.add_argument("--create-tables", action="store_true", help="Create necessary tables")
    parser.add_argument("--add-sample-data", action="store_true", help="Add sample data to the knowledge base")
    parser.add_argument("--update-embeddings", action="store_true", help="Update vector embeddings for all entities")
    parser.add_argument("--create-pg-vector", action="store_true", help="Create pgvector extension")
    parser.add_argument("--check-dependencies", action="store_true", help="Check if all required dependencies are installed")
    
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
    
    if args.create_pg_vector:
        create_pg_vector_extension()
    
    if args.create_tables:
        create_tables()
    
    if args.add_sample_data:
        add_sample_data()
    
    if args.update_embeddings:
        update_embeddings()
    
    if not (args.create_tables or args.add_sample_data or args.update_embeddings or args.create_pg_vector or args.check_dependencies):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())