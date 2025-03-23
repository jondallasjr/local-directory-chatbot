#!/usr/bin/env python3
"""
Supabase Admin Tool for Directory Chatbot

This script provides administrative functionality for managing the 
Supabase database, including:
1. Listing existing tables
2. Creating new tables
3. Modifying existing tables (add/remove/modify columns)
4. Defining and updating relationships between tables

Usage:
  python supabase_admin.py list-tables
  python supabase_admin.py create-table <table_name> <column_definitions>
  python supabase_admin.py add-column <table_name> <column_name> <column_type>
  python supabase_admin.py drop-column <table_name> <column_name>
  python supabase_admin.py create-relationship <table1> <column1> <table2> <column2> <relationship_type>
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any, Union
import logging
from dotenv import load_dotenv

from supabase import create_client, Client
from app.config.settings import SUPABASE_URL, SUPABASE_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Supabase client
def get_supabase_client() -> Client:
    """Get the Supabase client."""
    # Load environment variables
    load_dotenv()
    
    # Initialize Supabase client
    supabase_url = SUPABASE_URL
    supabase_key = SUPABASE_KEY
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or key not found. Check your .env file.")
        sys.exit(1)
    
    return create_client(supabase_url, supabase_key)

# Table Management Functions
def list_tables() -> List[str]:
    """List all tables in the database."""
    try:
        supabase = get_supabase_client()
        # Query information_schema.tables to get table names
        response = supabase.table("information_schema.tables")\
            .select("table_name")\
            .eq("table_schema", "public")\
            .execute()
        
        tables = [table["table_name"] for table in response.data]
        
        # Also list RLS policies
        for table in tables:
            try:
                policy_response = supabase.rpc(
                    "get_policies_for_table", 
                    {"table_name": table}
                ).execute()
                
                if policy_response.data:
                    policies = policy_response.data
                    logger.info(f"Policies for table '{table}':")
                    for policy in policies:
                        logger.info(f"  - {policy['policyname']}: {policy['permissive']} {policy['cmd']} ({policy['qual']})")
            except Exception as e:
                logger.warning(f"Could not fetch policies for table '{table}': {e}")
        
        return tables
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return []

def get_table_structure(table_name: str) -> Dict:
    """Get the structure of a specific table."""
    try:
        supabase = get_supabase_client()
        # Query information_schema.columns to get column information
        response = supabase.table("information_schema.columns")\
            .select("column_name,data_type,is_nullable,column_default")\
            .eq("table_schema", "public")\
            .eq("table_name", table_name)\
            .execute()
        
        return {
            "table_name": table_name,
            "columns": response.data
        }
    except Exception as e:
        logger.error(f"Error getting table structure for {table_name}: {e}")
        return {"table_name": table_name, "columns": [], "error": str(e)}

def create_table(table_name: str, column_definitions: Dict[str, str]) -> bool:
    """
    Create a new table with the specified columns.
    
    Args:
        table_name: Name of the table to create
        column_definitions: Dictionary mapping column names to their SQL definitions
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Build the SQL statement
        columns_sql = []
        for column_name, column_def in column_definitions.items():
            columns_sql.append(f'"{column_name}" {column_def}')
        
        # Always add id column as primary key if not specified
        if "id" not in column_definitions:
            columns_sql.insert(0, '"id" uuid PRIMARY KEY DEFAULT uuid_generate_v4()')
        
        # Always add timestamps if not specified
        if "created_at" not in column_definitions:
            columns_sql.append('"created_at" timestamptz NOT NULL DEFAULT now()')
        
        columns_str = ", ".join(columns_sql)
        sql = f'CREATE TABLE "{table_name}" ({columns_str});'
        
        # Execute the SQL
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        # Add RLS policy to allow authenticated users to read
        enable_rls_sql = f'ALTER TABLE "{table_name}" ENABLE ROW LEVEL SECURITY;'
        supabase.rpc("execute_sql", {"sql": enable_rls_sql}).execute()
        
        # Create a policy for authenticated users to read
        policy_sql = f"""
        CREATE POLICY "Enable read access for authenticated users" 
        ON "{table_name}" FOR SELECT 
        USING (auth.role() = 'authenticated');
        """
        supabase.rpc("execute_sql", {"sql": policy_sql}).execute()
        
        logger.info(f"Table '{table_name}' created successfully.")
        logger.info(f"Row-level security enabled with read policy for authenticated users.")
        return True
    except Exception as e:
        logger.error(f"Error creating table {table_name}: {e}")
        return False

def add_column(table_name: str, column_name: str, column_type: str) -> bool:
    """
    Add a column to an existing table.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column to add
        column_type: SQL type definition for the column
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Build and execute the SQL
        sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type};'
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        logger.info(f"Column '{column_name}' added to table '{table_name}' successfully.")
        return True
    except Exception as e:
        logger.error(f"Error adding column {column_name} to {table_name}: {e}")
        return False

def drop_column(table_name: str, column_name: str) -> bool:
    """
    Remove a column from an existing table.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column to remove
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Build and execute the SQL
        sql = f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}";'
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        logger.info(f"Column '{column_name}' dropped from table '{table_name}' successfully.")
        return True
    except Exception as e:
        logger.error(f"Error dropping column {column_name} from {table_name}: {e}")
        return False

def alter_column(table_name: str, column_name: str, new_definition: str) -> bool:
    """
    Modify an existing column.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column to modify
        new_definition: New SQL type or constraint definition
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Build and execute the SQL
        sql = f'ALTER TABLE "{table_name}" ALTER COLUMN "{column_name}" {new_definition};'
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        logger.info(f"Column '{column_name}' in table '{table_name}' altered successfully.")
        return True
    except Exception as e:
        logger.error(f"Error altering column {column_name} in {table_name}: {e}")
        return False

# Relationship Management Functions
def create_relationship(table1: str, column1: str, table2: str, column2: str, 
                        relationship_type: str = "foreign_key") -> bool:
    """
    Create a relationship between two tables.
    
    Args:
        table1: First table name
        column1: Column in first table
        table2: Second table name
        column2: Column in second table
        relationship_type: Type of relationship (foreign_key, many_to_many)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        if relationship_type == "foreign_key":
            # Create a foreign key constraint
            constraint_name = f"fk_{table1}_{column1}_to_{table2}_{column2}"
            sql = f'''
            ALTER TABLE "{table1}" ADD CONSTRAINT "{constraint_name}"
            FOREIGN KEY ("{column1}") REFERENCES "{table2}" ("{column2}");
            '''
            response = supabase.rpc("execute_sql", {"sql": sql}).execute()
            
            logger.info(f"Foreign key relationship created: {table1}.{column1} -> {table2}.{column2}")
            return True
            
        elif relationship_type == "many_to_many":
            # Create a junction table
            junction_table = f"{table1}_{table2}_junction"
            
            # Check if junction table already exists
            tables = list_tables()
            if junction_table in tables:
                logger.warning(f"Junction table {junction_table} already exists.")
                return False
            
            # Create the junction table
            create_table_sql = f'''
            CREATE TABLE "{junction_table}" (
                "id" uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
                "{table1}_id" uuid NOT NULL REFERENCES "{table1}" (id) ON DELETE CASCADE,
                "{table2}_id" uuid NOT NULL REFERENCES "{table2}" (id) ON DELETE CASCADE,
                "created_at" timestamptz NOT NULL DEFAULT now(),
                UNIQUE ("{table1}_id", "{table2}_id")
            );
            '''
            supabase.rpc("execute_sql", {"sql": create_table_sql}).execute()
            
            # Enable RLS
            enable_rls_sql = f'ALTER TABLE "{junction_table}" ENABLE ROW LEVEL SECURITY;'
            supabase.rpc("execute_sql", {"sql": enable_rls_sql}).execute()
            
            # Create a policy for authenticated users to read
            policy_sql = f"""
            CREATE POLICY "Enable read access for authenticated users" 
            ON "{junction_table}" FOR SELECT 
            USING (auth.role() = 'authenticated');
            """
            supabase.rpc("execute_sql", {"sql": policy_sql}).execute()
            
            logger.info(f"Many-to-many relationship created via junction table: {junction_table}")
            return True
        else:
            logger.error(f"Unsupported relationship type: {relationship_type}")
            return False
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        return False

def drop_relationship(table: str, constraint_name: str) -> bool:
    """
    Drop a relationship constraint.
    
    Args:
        table: Table containing the constraint
        constraint_name: Name of the constraint to drop
    
    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        
        # Build and execute the SQL
        sql = f'ALTER TABLE "{table}" DROP CONSTRAINT "{constraint_name}";'
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        logger.info(f"Constraint '{constraint_name}' dropped from table '{table}' successfully.")
        return True
    except Exception as e:
        logger.error(f"Error dropping constraint {constraint_name} from {table}: {e}")
        return False

def list_relationships() -> List[Dict]:
    """List all relationships (foreign keys) in the database."""
    try:
        supabase = get_supabase_client()
        # Query information_schema to get foreign key relationships
        sql = """
        SELECT
            tc.table_schema, 
            tc.constraint_name, 
            tc.table_name, 
            kcu.column_name, 
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name 
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
        """
        response = supabase.rpc("execute_sql", {"sql": sql}).execute()
        
        return response.data if hasattr(response, 'data') else []
    except Exception as e:
        logger.error(f"Error listing relationships: {e}")
        return []

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Supabase Database Administration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List tables command
    list_parser = subparsers.add_parser("list-tables", help="List all tables in the database")
    
    # Get table structure command
    structure_parser = subparsers.add_parser("table-structure", help="Show structure of a specific table")
    structure_parser.add_argument("table_name", help="Name of the table")
    
    # Create table command
    create_parser = subparsers.add_parser("create-table", help="Create a new table")
    create_parser.add_argument("table_name", help="Name of the table")
    create_parser.add_argument("columns", help='JSON string of column definitions, e.g., \'{"name": "text NOT NULL", "age": "integer"}\'')
    
    # Add column command
    add_column_parser = subparsers.add_parser("add-column", help="Add a column to an existing table")
    add_column_parser.add_argument("table_name", help="Name of the table")
    add_column_parser.add_argument("column_name", help="Name of the column to add")
    add_column_parser.add_argument("column_type", help="SQL type definition for the column")
    
    # Drop column command
    drop_column_parser = subparsers.add_parser("drop-column", help="Remove a column from an existing table")
    drop_column_parser.add_argument("table_name", help="Name of the table")
    drop_column_parser.add_argument("column_name", help="Name of the column to remove")
    
    # Alter column command
    alter_column_parser = subparsers.add_parser("alter-column", help="Modify an existing column")
    alter_column_parser.add_argument("table_name", help="Name of the table")
    alter_column_parser.add_argument("column_name", help="Name of the column to modify")
    alter_column_parser.add_argument("new_definition", help="New SQL type or constraint definition")
    
    # Create relationship command
    relationship_parser = subparsers.add_parser("create-relationship", help="Create a relationship between tables")
    relationship_parser.add_argument("table1", help="First table name")
    relationship_parser.add_argument("column1", help="Column in first table")
    relationship_parser.add_argument("table2", help="Second table name")
    relationship_parser.add_argument("column2", help="Column in second table")
    relationship_parser.add_argument("--type", dest="relationship_type", 
                                     choices=["foreign_key", "many_to_many"], 
                                     default="foreign_key",
                                     help="Type of relationship to create")
    
    # Drop relationship command
    drop_rel_parser = subparsers.add_parser("drop-relationship", help="Drop a relationship constraint")
    drop_rel_parser.add_argument("table", help="Table containing the constraint")
    drop_rel_parser.add_argument("constraint_name", help="Name of the constraint to drop")
    
    # List relationships command
    list_rel_parser = subparsers.add_parser("list-relationships", help="List all relationships in the database")
    
    args = parser.parse_args()
    
    if args.command == "list-tables":
        tables = list_tables()
        if tables:
            logger.info("Database tables:")
            for table in tables:
                logger.info(f"  - {table}")
        else:
            logger.info("No tables found or an error occurred.")
    
    elif args.command == "table-structure":
        structure = get_table_structure(args.table_name)
        if structure.get("columns"):
            logger.info(f"Structure of table '{args.table_name}':")
            for column in structure["columns"]:
                nullable = "NULL" if column["is_nullable"] == "YES" else "NOT NULL"
                default = f" DEFAULT {column['column_default']}" if column["column_default"] else ""
                logger.info(f"  - {column['column_name']} ({column['data_type']}) {nullable}{default}")
        else:
            logger.info(f"No columns found for table '{args.table_name}' or an error occurred.")
    
    elif args.command == "create-table":
        try:
            column_definitions = json.loads(args.columns)
            success = create_table(args.table_name, column_definitions)
            if success:
                logger.info(f"Table '{args.table_name}' created successfully.")
            else:
                logger.error(f"Failed to create table '{args.table_name}'.")
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for columns. Please provide a valid JSON string.")
    
    elif args.command == "add-column":
        success = add_column(args.table_name, args.column_name, args.column_type)
        if success:
            logger.info(f"Column '{args.column_name}' added to table '{args.table_name}' successfully.")
        else:
            logger.error(f"Failed to add column '{args.column_name}' to table '{args.table_name}'.")
    
    elif args.command == "drop-column":
        success = drop_column(args.table_name, args.column_name)
        if success:
            logger.info(f"Column '{args.column_name}' dropped from table '{args.table_name}' successfully.")
        else:
            logger.error(f"Failed to drop column '{args.column_name}' from table '{args.table_name}'.")
    
    elif args.command == "alter-column":
        success = alter_column(args.table_name, args.column_name, args.new_definition)
        if success:
            logger.info(f"Column '{args.column_name}' in table '{args.table_name}' altered successfully.")
        else:
            logger.error(f"Failed to alter column '{args.column_name}' in table '{args.table_name}'.")
    
    elif args.command == "create-relationship":
        success = create_relationship(args.table1, args.column1, args.table2, args.column2, args.relationship_type)
        if success:
            logger.info(f"Relationship created successfully.")
        else:
            logger.error(f"Failed to create relationship.")
    
    elif args.command == "drop-relationship":
        success = drop_relationship(args.table, args.constraint_name)
        if success:
            logger.info(f"Relationship constraint '{args.constraint_name}' dropped successfully.")
        else:
            logger.error(f"Failed to drop relationship constraint '{args.constraint_name}'.")
    
    elif args.command == "list-relationships":
        relationships = list_relationships()
        if relationships:
            logger.info("Database relationships:")
            for rel in relationships:
                logger.info(f"  - {rel['table_name']}.{rel['column_name']} -> {rel['foreign_table_name']}.{rel['foreign_column_name']} (constraint: {rel['constraint_name']})")
        else:
            logger.info("No relationships found or an error occurred.")
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())