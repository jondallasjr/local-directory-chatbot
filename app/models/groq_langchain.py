# app/models/groq_langchain.py

"""
LangChain integration module for Groq API.

This module provides classes and functions for interacting with Groq's LLM models
through LangChain, with support for prompt management and response parsing.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate

from app.config.settings import GROQ_API_KEY, LLM_MODEL, DEBUG

# Configure logging
logger = logging.getLogger("groq_langchain")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

class GroqChatCompletion:
    """
    LangChain-based chat completion handler for Groq API.
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = LLM_MODEL):
        """
        Initialize the chat completion handler.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self._llm = None
        
    @property
    def llm(self):
        """
        Lazy initialization of LLM to handle potential connection issues.
        """
        if self._llm is None:
            try:
                self._llm = ChatGroq(
                    api_key=self.api_key,
                    model_name=self.model,
                    temperature=0.7,
                )
                logger.info(f"Initialized Groq LLM with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq LLM: {str(e)}")
                # We'll handle the None case in the completion method
        return self._llm
    
    def chat_completion(self, 
                      messages: List[Dict[str, str]], 
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a chat completion using the Groq API via LangChain.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override the default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completion result with response text and usage info
        """
        if self.llm is None:
            logger.error("LLM not initialized, cannot generate completion")
            return {
                "error": "LLM initialization failed",
                "text": "Sorry, I'm experiencing connection issues. Please try again later."
            }
        
        try:
            # Convert dictionary messages to LangChain message objects
            lc_messages = []
            for msg in messages:
                role = msg.get('role', 'user').lower()
                content = msg.get('content', '')
                
                if role == 'system':
                    lc_messages.append(SystemMessage(content=content))
                elif role == 'assistant':
                    lc_messages.append(AIMessage(content=content))
                else:  # Default to user message
                    lc_messages.append(HumanMessage(content=content))
            
            # Set parameters for the completion
            params = {}
            if temperature is not None:
                params['temperature'] = temperature
            if max_tokens is not None:
                params['max_tokens'] = max_tokens
            
            # Log the request
            logger.debug(f"Sending request to Groq with {len(lc_messages)} messages")
            for i, msg in enumerate(lc_messages):
                logger.debug(f"Message {i+1} ({msg.type}): {msg.content[:100]}...")
            
            # Generate completion
            start_time = __import__('time').time()
            response = self.llm.invoke(lc_messages, **params)
            elapsed = __import__('time').time() - start_time
            
            # Extract response text
            response_text = response.content
            
            # Log the response
            logger.debug(f"Received response in {elapsed:.2f}s: {response_text[:100]}...")
            
            return {
                "text": response_text,
                "elapsed_seconds": elapsed,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            return {
                "error": str(e),
                "text": "Sorry, I encountered an error while generating a response."
            }
    
    def template_completion(self, 
                          template: str, 
                          variables: Dict[str, str],
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a completion using a template with variables.
        
        Args:
            template: Prompt template with variable placeholders
            variables: Dictionary of variables to fill the template
            temperature: Override the default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Completion result with response text and usage info
        """
        try:
            # Create ChatPromptTemplate
            prompt_template = ChatPromptTemplate.from_template(template)
            
            # Format the prompt with the variables
            messages = prompt_template.format_messages(**variables)
            
            # Log the formatted prompt
            logger.debug(f"Formatted template with {len(variables)} variables:")
            for i, msg in enumerate(messages):
                logger.debug(f"Message {i+1} ({msg.type}): {msg.content[:100]}...")
            
            # Set parameters for the completion
            params = {}
            if temperature is not None:
                params['temperature'] = temperature
            if max_tokens is not None:
                params['max_tokens'] = max_tokens
            
            # Generate completion
            start_time = __import__('time').time()
            response = self.llm.invoke(messages, **params)
            elapsed = __import__('time').time() - start_time
            
            # Extract response text
            response_text = response.content
            
            # Log the response
            logger.debug(f"Received template response in {elapsed:.2f}s: {response_text[:100]}...")
            
            return {
                "text": response_text,
                "elapsed_seconds": elapsed,
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Error generating template completion: {str(e)}")
            return {
                "error": str(e),
                "text": "Sorry, I encountered an error while processing your request."
            }
    
    def parse_json_response(self, completion_text: str) -> Dict[str, Any]:
        """
        Parse a JSON response from completion text.
        
        Args:
            completion_text: Raw completion text that should contain JSON
            
        Returns:
            Parsed JSON as a dictionary, or error dictionary
        """
        try:
            # Try to find JSON object in the response (it might be surrounded by text)
            start_idx = completion_text.find('{')
            end_idx = completion_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = completion_text[start_idx:end_idx]
                return json.loads(json_str)
            
            # If we couldn't find JSON brackets, try to parse the whole text
            return json.loads(completion_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {str(e)}")
            logger.debug(f"Response text: {completion_text}")
            return {
                "error": "Failed to parse JSON response",
                "raw_text": completion_text
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {str(e)}")
            return {
                "error": str(e),
                "raw_text": completion_text
            }


# Singleton instance for reuse
@lru_cache(maxsize=1)
def get_groq_client() -> GroqChatCompletion:
    """
    Get a singleton instance of the GroqChatCompletion class.
    
    Returns:
        GroqChatCompletion instance
    """
    return GroqChatCompletion()


# Simple test function to verify the module is working
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    client = get_groq_client()
    
    # Test simple completion
    completion = client.chat_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"}
    ])
    
    print(f"Completion result: {completion}")
    
    # Test template completion
    template = """
    You are a helpful assistant for a community directory.
    
    The user has sent the following message:
    {user_message}
    
    Please respond in a helpful way.
    """
    
    template_result = client.template_completion(
        template=template,
        variables={"user_message": "Can you tell me about local events this weekend?"}
    )
    
    print(f"Template result: {template_result}")
    
    # Test JSON parsing
    json_template = """
    You will respond with a JSON object containing the following fields:
    - greeting: A friendly greeting
    - information: Some information about community directories
    
    Make sure your response is valid JSON.
    """
    
    json_result = client.template_completion(
        template=json_template,
        variables={},
        temperature=0.2
    )
    
    parsed = client.parse_json_response(json_result["text"])
    print(f"Parsed JSON: {parsed}")