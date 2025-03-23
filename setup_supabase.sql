-- This script sets up the Supabase database for the graph knowledge base
-- Run this in the Supabase SQL Editor

-- First, enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create the pgvector extension (we've verified this exists)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create entity table first, without the vector column
CREATE TABLE IF NOT EXISTS "entities" (
    "id" uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    "entity_type" text NOT NULL,
    "entity_name" text NOT NULL,
    "attributes" jsonb DEFAULT '{}'::jsonb,
    "created_by" uuid,
    "verification_status" text DEFAULT 'unverified',
    "metadata" jsonb DEFAULT '{}'::jsonb,
    "tags" text[] DEFAULT '{}'::text[],
    "created_at" timestamptz DEFAULT now(),
    "updated_at" timestamptz DEFAULT now()
);

-- Add the vector column separately
ALTER TABLE "entities" 
ADD COLUMN IF NOT EXISTS "vector_embedding" vector(384);

-- Create entity relationships table
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

-- Create indices for faster queries
CREATE INDEX IF NOT EXISTS "entities_entity_type_idx" ON "entities" ("entity_type");
CREATE INDEX IF NOT EXISTS "entities_entity_name_idx" ON "entities" ("entity_name" text_pattern_ops);
CREATE INDEX IF NOT EXISTS "entity_relationships_from_idx" ON "entity_relationships" ("from_entity_id");
CREATE INDEX IF NOT EXISTS "entity_relationships_to_idx" ON "entity_relationships" ("to_entity_id");
CREATE INDEX IF NOT EXISTS "entity_relationships_type_idx" ON "entity_relationships" ("relationship_type");

-- Create vector index AFTER the column exists
CREATE INDEX IF NOT EXISTS "entities_vector_idx"
ON "entities"
USING ivfflat (vector_embedding vector_cosine_ops)
WITH (lists = 100);

-- Create the execute_sql function for database operations
CREATE OR REPLACE FUNCTION execute_sql(sql text, params jsonb DEFAULT '[]'::jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result jsonb;
    param_values text[];
    i int;
BEGIN
    -- Extract parameter values as an array
    IF jsonb_array_length(params) > 0 THEN
        param_values := array_fill(NULL::text, ARRAY[jsonb_array_length(params)]);
        FOR i IN 0..jsonb_array_length(params)-1 LOOP
            param_values[i+1] := params->i;
        END LOOP;
    END IF;

    -- Execute the SQL with parameters
    EXECUTE sql INTO result USING param_values;

    -- Return the result
    RETURN result;
EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
        'error', SQLERRM,
        'detail', SQLSTATE,
        'sql', sql
    );
END;
$$;

-- Create function to add entity with embedding
CREATE OR REPLACE FUNCTION add_entity(
    p_entity_type text,
    p_entity_name text,
    p_attributes jsonb DEFAULT '{}'::jsonb,
    p_created_by uuid DEFAULT NULL,
    p_tags text[] DEFAULT '{}'::text[],
    p_vector_embedding vector(384) DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    entity_id uuid;
BEGIN
    INSERT INTO entities (
        entity_type,
        entity_name,
        attributes,
        created_by,
        tags,
        vector_embedding
    ) VALUES (
        p_entity_type,
        p_entity_name,
        p_attributes,
        p_created_by,
        p_tags,
        p_vector_embedding
    )
    RETURNING id INTO entity_id;

    RETURN entity_id;
END;
$$;

-- Create function to add relationship
CREATE OR REPLACE FUNCTION add_relationship(
    p_from_entity_id uuid,
    p_to_entity_id uuid,
    p_relationship_type text,
    p_attributes jsonb DEFAULT '{}'::jsonb,
    p_weight real DEFAULT 1.0
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    relationship_id uuid;
BEGIN
    INSERT INTO entity_relationships (
        from_entity_id,
        to_entity_id,
        relationship_type,
        attributes,
        weight
    ) VALUES (
        p_from_entity_id,
        p_to_entity_id,
        p_relationship_type,
        p_attributes,
        p_weight
    )
    RETURNING id INTO relationship_id;

    RETURN relationship_id;
END;
$$;

-- Create search function that combines vector search with filters
CREATE OR REPLACE FUNCTION search_entities(
    p_search_text text,
    p_entity_types text[] DEFAULT NULL,
    p_tags text[] DEFAULT NULL,
    p_limit int DEFAULT 10,
    p_vector_embedding vector(384) DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    entity_type text,
    entity_name text,
    attributes jsonb,
    tags text[],
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- With vector search
    IF p_vector_embedding IS NOT NULL THEN
        RETURN QUERY
        SELECT
            e.id,
            e.entity_type,
            e.entity_name,
            e.attributes,
            e.tags,
            1 - (e.vector_embedding <=> p_vector_embedding) AS similarity
        FROM entities e
        WHERE
            (p_entity_types IS NULL OR e.entity_type = ANY(p_entity_types))
            AND (p_tags IS NULL OR e.tags && p_tags)
            AND e.vector_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT p_limit;
    -- Text-based search
    ELSE
        RETURN QUERY
        SELECT
            e.id,
            e.entity_type,
            e.entity_name,
            e.attributes,
            e.tags,
            0::float AS similarity
        FROM entities e
        WHERE
            (p_entity_types IS NULL OR e.entity_type = ANY(p_entity_types))
            AND (p_tags IS NULL OR e.tags && p_tags)
            AND (p_search_text IS NULL OR e.entity_name ILIKE '%' || p_search_text || '%')
        ORDER BY e.entity_name
        LIMIT p_limit;
    END IF;
END;
$$;

-- Create function to get entity with its relationships
CREATE OR REPLACE FUNCTION get_entity_with_relationships(
    p_entity_id uuid,
    p_max_depth int DEFAULT 1,
    p_max_relationships_per_entity int DEFAULT 5
)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    entity_data jsonb;
    relationships jsonb;
    related_entities jsonb;
BEGIN
    -- Get the main entity
    SELECT jsonb_build_object(
        'id', e.id,
        'entity_type', e.entity_type,
        'entity_name', e.entity_name,
        'attributes', e.attributes,
        'tags', e.tags,
        'verification_status', e.verification_status,
        'created_at', e.created_at
    )
    INTO entity_data
    FROM entities e
    WHERE e.id = p_entity_id;

    IF entity_data IS NULL THEN
        RETURN jsonb_build_object('error', 'Entity not found');
    END IF;

    -- Get outgoing relationships
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', r.id,
            'relationship_type', r.relationship_type,
            'direction', 'outgoing',
            'target_entity', jsonb_build_object(
                'id', e.id,
                'entity_type', e.entity_type,
                'entity_name', e.entity_name
            ),
            'attributes', r.attributes,
            'weight', r.weight
        )
    )
    INTO relationships
    FROM entity_relationships r
    JOIN entities e ON r.to_entity_id = e.id
    WHERE r.from_entity_id = p_entity_id
    ORDER BY r.weight DESC
    LIMIT p_max_relationships_per_entity;

    -- Add incoming relationships
    WITH incoming AS (
        SELECT
            r.id,
            r.relationship_type,
            r.attributes,
            r.weight,
            e.id as source_id,
            e.entity_type as source_type,
            e.entity_name as source_name
        FROM entity_relationships r
        JOIN entities e ON r.from_entity_id = e.id
        WHERE r.to_entity_id = p_entity_id
        ORDER BY r.weight DESC
        LIMIT p_max_relationships_per_entity
    )
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', i.id,
            'relationship_type', i.relationship_type,
            'direction', 'incoming',
            'source_entity', jsonb_build_object(
                'id', i.source_id,
                'entity_type', i.source_type,
                'entity_name', i.source_name
            ),
            'attributes', i.attributes,
            'weight', i.weight
        )
    )
    INTO related_entities
    FROM incoming i;

    -- Combine the results
    relationships := COALESCE(relationships, '[]'::jsonb) || COALESCE(related_entities, '[]'::jsonb);

    -- Add relationships to the entity data
    entity_data := entity_data || jsonb_build_object('relationships', relationships);

    RETURN entity_data;
END;
$$;

-- Create trigger to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_entities_timestamp
BEFORE UPDATE ON entities
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_relationships_timestamp
BEFORE UPDATE ON entity_relationships
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- RLS Policies for secure access
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_relationships ENABLE ROW LEVEL SECURITY;

-- Allow anonymous read access to entities and relationships
CREATE POLICY "Allow anonymous read access to entities"
    ON entities FOR SELECT
    USING (true);

CREATE POLICY "Allow anonymous read access to relationships"
    ON entity_relationships FOR SELECT
    USING (true);

-- Allow authenticated users to create entities and relationships
CREATE POLICY "Allow authenticated users to create entities"
    ON entities FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated users to create relationships"
    ON entity_relationships FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow users to update their own entities
CREATE POLICY "Allow users to update their own entities"
    ON entities FOR UPDATE
    TO authenticated
    USING (auth.uid() = created_by)
    WITH CHECK (auth.uid() = created_by);

-- Enable the Supabase GraphQL API for these tables (optional)
COMMENT ON TABLE entities IS E'@graphql({"name": "Entity"})';
COMMENT ON TABLE entity_relationships IS E'@graphql({"name": "EntityRelationship"})';