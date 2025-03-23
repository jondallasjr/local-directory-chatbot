# Database Migrations

This directory contains SQL migration files for setting up and maintaining the database schema for the Local Directory Chatbot.

## Migration Files

- `001_setup_graph_kb.sql`: Initial setup for the Graph Knowledge Base including entity and relationship tables, necessary extensions, and helper functions.

## How to Apply Migrations

These SQL files should be executed in order in the Supabase SQL Editor. 

1. Log in to your Supabase dashboard
2. Navigate to the SQL Editor
3. Open the migration file (e.g., `001_setup_graph_kb.sql`)
4. Execute the SQL script

## Migration File Naming Convention

Migration files follow this naming pattern:

```
NNN_description.sql
```

Where:
- `NNN` is a sequential number (001, 002, etc.) to ensure migrations are applied in the correct order
- `description` is a brief description of what the migration does

## Development Workflow

When making database schema changes:

1. Create a new migration file with a sequential number
2. Test the migration in a development environment
3. Add the migration file to version control
4. Apply the migration to production when ready

## Important Notes

- Never modify existing migration files after they've been applied to production
- Create new migration files for any changes to the database schema
- Always test migrations in a development environment before applying to production