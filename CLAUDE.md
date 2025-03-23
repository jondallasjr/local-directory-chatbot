# Directory Chatbot Development Guidelines

## Commands
- Run API server: `python main.py api [--reload]`
- Run WhatsApp simulator: `python main.py simulator`
- Run both together: `python main.py both [--reload]`
- Test API connection: `python api_test.py`
- Test LLM integration: `python llm_test.py`

## Code Style
- **Imports**: Group in order: standard library, third-party, local app imports
- **Typing**: Use type hints for function parameters and return values
- **Docstrings**: Include descriptive docstrings for modules, classes, and functions
- **Error Handling**: Use try/except with specific exception types; log errors with context
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Logging**: Use app.utils.logging_utils for consistent logging
- **Structure**: Follow the established modular architecture: api/, models/, utils/, workflows/

## Database & LLM
- Supabase operations through supabase_client.py
- LLM operations through models/llm.py
- Always use error handling for async operations