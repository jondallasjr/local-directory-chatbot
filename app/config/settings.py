"""
Settings module for the application.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv

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