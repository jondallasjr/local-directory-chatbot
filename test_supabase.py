from app.database.supabase_client import supabase
import sys

def test_execute_sql():
    try:
        sql = "SELECT schema_name FROM information_schema.schemata"
        result = supabase.rpc("execute_sql", {"sql": sql}).execute()
        print(f"Execute SQL result: {result.data}")
        return True
    except Exception as e:
        print(f"Execute SQL error: {e}")
        return False

def test_create_extension():
    try:
        sql = "CREATE EXTENSION IF NOT EXISTS vector"
        result = supabase.rpc("execute_sql", {"sql": sql}).execute()
        print(f"Create extension result: {result.data}")
        return True
    except Exception as e:
        print(f"Create extension error: {e}")
        return False

def test_basic_query():
    try:
        result = supabase.table("users").select("*").limit(1).execute()
        print(f"Basic query result: {result.data}")
        return True
    except Exception as e:
        print(f"Basic query error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Supabase connection...")
    print("\n1. Basic query test:")
    test_basic_query()
    
    print("\n2. Execute SQL test:")
    test_execute_sql()
    
    print("\n3. Create extension test:")
    test_create_extension()
