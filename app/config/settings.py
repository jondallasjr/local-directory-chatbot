"""
Settings module for the application.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv
import logging
import traceback
from functools import wraps
import inspect
import sys

# Load environment variables from .env file
load_dotenv()

# LLM Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama3-8b-8192"  # Llama 3 8B model on Groq

# Database Settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Messaging Settings
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Application Settings
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Directory Chatbot Settings
DEFAULT_WORKFLOW = "determine_workflow"  # Default entry point workflow
MAX_CONTEXT_MESSAGES = 10  # Maximum number of messages to include in context

def setup_enhanced_logging():
    """Configure enhanced logging with file, class, and function information."""
    
    # Define a custom formatter that includes more details
    class DetailedFormatter(logging.Formatter):
        def formatException(self, exc_info):
            """Format exception information with detailed traceback."""
            result = super().formatException(exc_info)
            return f"{result}\n\nDetailed traceback:\n{traceback.format_exc()}"
        
        def format(self, record):
            """Add file, class, and function details to the log record."""
            # Add the file path if not present
            if not hasattr(record, 'pathname') or not record.pathname:
                record.pathname = 'unknown'
                
            # Extract filename from path
            record.filename = record.pathname.split('/')[-1]
            
            # Format the log message with details
            return f"[{record.levelname}] {record.filename}:{record.lineno} - {record.funcName}() - {record.getMessage()}"
    
    # Configure the root logger
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DetailedFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)  # Set the desired log level
    
    # Set propagate=False for other handlers to avoid duplicate logs
    for handler in root_logger.handlers:
        if handler != handler:
            handler.setLevel(logging.WARNING)
            
    return root_logger

# Call this function to set up logging
logger = setup_enhanced_logging()