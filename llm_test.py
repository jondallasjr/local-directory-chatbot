from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key available: {'Yes' if api_key else 'No'}")

# Initialize LLM
try:
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192"
    )
    
    # Test simple completion
    response = llm.invoke("Hello, can you respond with valid JSON? Please respond with: {\"status\": \"ok\"}")
    print("LLM Response:", response.content)
    print("Test successful!")
except Exception as e:
    print(f"Error testing LLM: {str(e)}")