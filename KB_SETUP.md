# Graph Knowledge Base Setup Guide

This guide will help you set up the Graph Knowledge Base for the Local Directory Chatbot application.

## Prerequisites

1. A Supabase account and project
2. Python 3.9+ installed
3. Required Python dependencies (see below)

## Required Dependencies

The Graph Knowledge Base requires the following Python packages:

```bash
# Core dependencies
pip install supabase
pip install langchain-community
pip install numpy

# For embeddings
pip install sentence-transformers
pip install torch
pip install transformers
pip install huggingface-hub

# Optional but recommended
pip install pgvector  # Python client for pgvector
```

You can install all dependencies at once using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Database Setup

The Graph Knowledge Base requires a properly configured Supabase database with the pgvector extension and necessary tables and functions.

### 1. Run the SQL Setup Script

1. Log in to your Supabase dashboard
2. Navigate to the SQL Editor
3. Copy the contents of `setup_supabase.sql` (or upload the file)
4. Run the SQL script to create:
   - The pgvector extension
   - The required tables (entities and entity_relationships)
   - Helper functions for entity and relationship management
   - Search functions with vector similarity

The script has been designed to run safely, even if parts of it have already been executed before.

## Verifying Your Setup

After running the SQL setup script, you can verify that everything is working correctly by running the provided test script:

```bash
python test_graph_kb.py
```

This script will:

1. Test the connection to your Supabase database
2. Verify the execute_sql function works
3. Add some test entities with embeddings
4. Test searching for entities using both text and vector search
5. Clean up the test data

## Common Issues and Solutions

### Issue: "Column vector_embedding does not exist"

This can happen if:
- The pgvector extension wasn't properly installed
- You're trying to access the vector_embedding column before it's created

**Solution:**
1. Verify the pgvector extension is installed:
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```
2. Modify your entity creation to add the vector column separately:
```sql
ALTER TABLE "entities" ADD COLUMN IF NOT EXISTS "vector_embedding" vector(384);
```

### Issue: "Could not find the function public.execute_sql(sql)"

This happens when the execute_sql function isn't defined in your database.

**Solution:**
Run the execute_sql function creation part of the setup script:
```sql
CREATE OR REPLACE FUNCTION execute_sql(sql text, params jsonb DEFAULT '[]'::jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
-- Function body as in setup_supabase.sql
$$;
```

### Issue: Missing sentence-transformers or related packages

**Solution:**
Install the required packages:
```bash
pip install sentence-transformers torch transformers huggingface-hub
```

## Using the Graph Knowledge Base

### Adding Entities

```python
from app.database.graph_kb import GraphKnowledgeBase

# Initialize the KB
kb = GraphKnowledgeBase()

# Add a business entity
business_id = kb.add_entity(
    entity_type="business",
    entity_name="ABC Plumbing",
    attributes={
        "description": "Professional plumbing services",
        "phone": "+27731234567",
        "address": "123 Main St, Johannesburg"
    },
    tags=["plumbing", "services", "repair"]
)

# Add a product entity
product_id = kb.add_entity(
    entity_type="product",
    entity_name="Organic Tomatoes",
    attributes={
        "description": "Fresh organic tomatoes",
        "price": "R25 per kg",
        "available": True
    },
    tags=["produce", "vegetable", "organic"]
)

# Create a relationship between them
kb.add_relationship(
    from_entity_id=business_id,
    to_entity_id=product_id,
    relationship_type="sells",
    attributes={"notes": "Regular supplier"},
    weight=0.8
)
```

### Searching Entities

```python
# Text-based search
results = kb.search_entities(
    search_text="plumbing",
    entity_types=["business"],
    limit=5
)

# Vector similarity search
results = kb.search_entities(
    search_text="professional repair services",
    entity_types=["business", "service"],
    use_vector_search=True,
    limit=10
)

# Get an entity with its relationships
entity_with_rels = kb.get_entity_with_relationships(
    entity_id=business_id,
    max_depth=2,
    max_relationships_per_entity=10
)
```

## Integration with the WhatsApp Directory Chatbot

The Graph Knowledge Base is integrated with the WhatsApp Directory Chatbot through workflow handlers. When a user sends a message asking about businesses, products, or services, the following happens:

1. The ActionHandler receives the message
2. The LLM determines the user's intent (search, add, update)
3. For search queries, the GraphKnowledgeBase.search_entities method is called
4. Search results are formatted and sent back to the user

### Example Search Workflow

```python
# In a workflow handler
async def handle_search_query(context, message):
    kb = GraphKnowledgeBase()
    
    # Extract search parameters from the message using the LLM
    search_params = await extract_search_params(message)
    
    # Search for entities
    results = kb.search_entities(
        search_text=search_params["query"],
        entity_types=search_params["types"],
        tags=search_params.get("tags"),
        use_vector_search=True
    )
    
    # Format results for WhatsApp
    formatted_results = format_search_results(results)
    
    return formatted_results
```

## Advanced Topics

### Custom Embeddings

You can use custom embedding models by modifying the `get_embedding_for_text` function in `app/models/vector_search.py`:

```python
def get_embedding_for_text(text, model=None):
    """Get embedding for text using the specified model or default"""
    if model is None:
        # Initialize the default model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embedding
    embedding = model.encode(text)
    return embedding
```

### Graph Traversal

For more complex relationship queries, you can use the `traverse_graph` method:

```python
# Find all businesses that sell organic products
results = kb.traverse_graph(
    start_entity_type="product",
    start_entity_filter={"tags": ["organic"]},
    relationship_type="sells",
    relationship_direction="incoming",
    target_entity_type="business",
    max_depth=1
)
```

## Troubleshooting

If you encounter issues with the Graph Knowledge Base, check the following:

1. Database connection: Verify your Supabase URL and key in the .env file
2. pgvector extension: Make sure it's properly installed
3. Embeddings: Check that sentence-transformers is installed and working
4. execute_sql function: Verify it exists in your database
5. Tables: Confirm that entities and entity_relationships tables exist

Run the test script with detailed logging for more information:

```bash
python -m app.database.test_graph_kb --verbose
```

## Contributing

If you want to enhance the Graph Knowledge Base, consider the following areas:

1. Additional relationship types for different directory entities
2. Improved vector search algorithms
3. Caching mechanisms for frequently accessed entities
4. Enhanced security and access control
5. Integration with external data sources

Please follow the project's contribution guidelines when submitting changes.