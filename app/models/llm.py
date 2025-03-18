"""
Enhanced LLM integration module with comprehensive logging.

This module enhances the existing LLM handler with detailed logging
of all requests and responses, plus improved error handling.
"""

import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any, Union
import traceback

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.messages import BaseMessage

from app.config.settings import GROQ_API_KEY, LLM_MODEL, DEBUG

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("llm")

# Custom LLM wrapper to intercept and log all communications
class LoggingChatGroq(ChatGroq):
    """
    Extension of ChatGroq that logs all requests and responses.
    """
    
    def invoke(self, input: Union[str, List[BaseMessage]], **kwargs) -> Any:
        """Override invoke to add logging."""
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Log the request
        if isinstance(input, str):
            logger.debug(f"[{request_id}] LLM REQUEST (text):\n{input}\n")
        else:
            logger.debug(f"[{request_id}] LLM REQUEST (messages):")
            for i, msg in enumerate(input):
                logger.debug(f"[{request_id}] Message {i+1} ({msg.type}): {msg.content}")
        
        logger.debug(f"[{request_id}] LLM PARAMETERS: {kwargs}")
        
        try:
            # Call the parent class method
            start_time = time.time()
            response = super().invoke(input, **kwargs)
            elapsed = time.time() - start_time
            
            # Log the response
            if hasattr(response, 'content'):
                logger.debug(f"[{request_id}] LLM RESPONSE ({elapsed:.2f}s):\n{response.content}\n")
            else:
                logger.debug(f"[{request_id}] LLM RESPONSE ({elapsed:.2f}s):\n{response}\n")
            
            return response
        except Exception as e:
            logger.error(f"[{request_id}] LLM ERROR: {str(e)}")
            logger.error(f"[{request_id}] TRACEBACK: {traceback.format_exc()}")
            raise

# Initialize the LLM with improved logging
try:
    llm = LoggingChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.7,  # Added explicit temperature
    )
    logger.info(f"LLM initialized: {LLM_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    llm = None

class PromptTemplates:
    """
    Collection of prompt templates used in the system.
    """
    
    ACTION_DETERMINATION = """
Determine what the next ACTION should be.

<USER DETAILS>

<USER ENTITY JSON>
{user_entity_json}
</USER ENTITY JSON>

<ACTIVE CONTEXT>
{active_context}
</ACTIVE CONTEXT>

<CURRENT WORKFLOW>
Workflow Name:
{workflow_name}

Workflow Steps:
{workflow_steps}
</CURRENT WORKFLOW>

<WORKFLOWS>
{available_workflows}
</WORKFLOWS>

<MESSAGES>
{messages}
</MESSAGES>

</USER DETAILS>

<ACTION TYPES>
{action_types}
</ACTION TYPES>

Respond in perfectly formatted JSON with keys "Thinking", "Uncertainty", "Step", and "Next Action".

The value for Thinking should be thinking aloud, choosing the best next step and the rationale behind it. Write very concisely (5-10 words), unless technical specificity is required.

The value for Uncertainty can be left blank unless there is information missing making it difficult to determine the appropriate next Action.

The value for Step should be a concise text description of the particular step being executed according to the CURRENT WORKFLOW. The value should either be the same as the previous step, or the next step according to the CURRENT WORKFLOW.

The value for Next Action should be ONE item from the list of ACTION TYPES - the most appropriate next action to follow the ACTION HISTORY and LOGS in accordance with the WORKFLOW.

Start your response with a single opening brace and end it with a single closing brace.
"""

    MESSAGE_GENERATION = """
Write a message to the User following the WRITING GUIDELINES.

<USER DETAILS>

<USER ENTITY JSON>
{user_entity_json}
</USER ENTITY JSON>

<ACTIVE CONTEXT>
{active_context}
</ACTIVE CONTEXT>

<CURRENT WORKFLOW>
Workflow Name:
{workflow_name}

Workflow Steps:
{workflow_steps}
</CURRENT WORKFLOW>

<MESSAGES>
{messages}
</MESSAGES>

</USER DETAILS>

WRITING GUIDELINES:
1. Be concise and friendly.
2. Use natural, conversational language appropriate for WhatsApp.
3. Don't use complex formatting or long paragraphs - keep it mobile-friendly.
4. Be helpful and provide clear instructions or questions when needed.
5. If collecting information, ask one question at a time.
6. Include relevant details from the context when appropriate.

Respond in perfectly formatted JSON with keys 'Thinking' and 'Message'. Start your response with a single opening brace and end it with a single closing brace.
"""

    KNOWLEDGEBASE_UPSERT = """
Determine the appropriate JSON to upsert appropriate to the knowledgebase.

<USER ENTITY JSON>
{user_entity_json}
</USER ENTITY JSON>

<ACTIVE CONTEXT>
{active_context}
</ACTIVE CONTEXT>

<CURRENT WORKFLOW>
Workflow Name:
{workflow_name}

Workflow Steps:
{workflow_steps}
</CURRENT WORKFLOW>

<MESSAGES>
{messages}
</MESSAGES>

Respond in perfect JSON, with the following SCHEMA:

{
 "reasoning": "",
 "upsertJSON": {} // Based on what you see in the Active Context. Be sure to start at the root.
}
"""

    QUERY_GENERATION = """
Generate a database query based on the user's request.

<USER QUERY>
{user_query}
</USER QUERY>

<DATABASE SCHEMA>
{db_schema}
</DATABASE SCHEMA>

<ENTITY TYPES>
{entity_types}
</ENTITY TYPES>

Respond with the appropriate query in JSON format:

{
 "reasoning": "",
 "queryType": "", // "exact", "fuzzy", "vector", or "combined"
 "entityTypes": [], // Which entity types to search
 "filters": {}, // Key-value pairs for filtering
 "searchText": "", // Text to use for semantic search
 "limit": 10 // Number of results to return
}
"""

    CONTEXT_MANAGEMENT = """
Determine what context is relevant for the current conversation.

<CURRENT ACTIVE CONTEXT>
{current_context}
</CURRENT ACTIVE CONTEXT>

<RECENT MESSAGES>
{messages}
</RECENT MESSAGES>

<USER WORKFLOW>
{current_workflow}
</USER WORKFLOW>

Respond with updated context information:

{
 "reasoning": "",
 "keepContext": [], // List of context keys to keep
 "removeContext": [], // List of context keys to remove
 "addContext": {} // New context to add
}
"""


class LLMUtils:
    """
    Utility functions for LLM interactions.
    """
    
    @staticmethod
    def format_messages_history(messages: List[Dict]) -> str:
        """
        Format message history for prompt inclusion.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted message history string
        """
        messages_str = ""
        for msg in messages:
            direction = msg.get('direction', 'unknown')
            content = msg.get('message_content', '')
            timestamp = msg.get('timestamp', '')
            messages_str += f"{direction} [{timestamp}]: {content}\n"
        return messages_str
    
    @staticmethod
    def extract_json_from_response(content: str) -> str:
        """
        Extract JSON content from LLM response.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Extracted JSON content
        """
        # Log the raw content first
        logger.debug(f"Extracting JSON from raw content:\n{content}")
        
        # If content is already valid JSON, return it
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from the response
        try:
            # Look for JSON starting with { and ending with }
            content = content.strip()
            
            # Find all potential JSON objects (there might be text before or after)
            start_idx = content.find('{')
            if start_idx >= 0:
                # Track braces to find proper closing
                open_braces = 0
                for i in range(start_idx, len(content)):
                    if content[i] == '{':
                        open_braces += 1
                    elif content[i] == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            # Found complete JSON object
                            extracted = content[start_idx:i+1]
                            try:
                                # Validate it's valid JSON
                                json.loads(extracted)
                                logger.debug(f"Successfully extracted JSON: {extracted}")
                                return extracted
                            except json.JSONDecodeError:
                                logger.warning(f"Found JSON-like structure but it's invalid: {extracted}")
            
            logger.warning(f"No valid JSON found in content")
            return content
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return content
    
    @staticmethod
    def parse_json_response(content: str) -> Dict:
        """
        Parse JSON content from LLM response.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed JSON as dict
        """
        original_content = content
        logger.debug(f"Attempting to parse as JSON: {content[:500]}...")
        
        # Try to extract JSON from the response
        try:
            # First try to parse as is
            try:
                parsed = json.loads(content)
                logger.debug(f"Successfully parsed JSON directly")
                return parsed
            except json.JSONDecodeError:
                # Try to extract JSON
                extracted_json = LLMUtils.extract_json_from_response(content)
                
                if extracted_json != content:
                    logger.debug(f"Using extracted JSON: {extracted_json}")
                    try:
                        return json.loads(extracted_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Extracted JSON is invalid: {extracted_json}")
                
                # If extraction didn't work, try more aggressive methods
                # Look for the first { and last }
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        strict_extract = content[start_idx:end_idx]
                        logger.debug(f"Trying strict extract: {strict_extract}")
                        return json.loads(strict_extract)
                    except json.JSONDecodeError:
                        logger.warning(f"Strict extract is invalid JSON")
                
                # Last resort: try to build a JSON object from parts of the response
                if ":" in content and ("Thinking" in content or "Message" in content):
                    logger.debug("Attempting to construct JSON from key-value pairs")
                    # Try to extract key-value pairs
                    result = {}
                    
                    for line in content.split('\n'):
                        line = line.strip()
                        if ':' in line and not line.startswith('{') and not line.startswith('}'):
                            parts = line.split(':', 1)
                            key = parts[0].strip().strip('"')
                            value = parts[1].strip().strip(',').strip('"')
                            result[key] = value
                    
                    if result:
                        logger.debug(f"Constructed JSON from parts: {result}")
                        return result
                
                logger.error(f"Could not extract valid JSON from: {original_content}")
                raise ValueError("No JSON content found in response")
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Original content: {original_content}")
            
            # Return a fallback response instead of raising an exception
            return {
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_content": original_content[:500] + ("..." if len(original_content) > 500 else "")
            }
        
class PromptHandler:
    """
    Base class for handling LLM prompts and responses.
    """
    
    def __init__(self, template: str):
        """
        Initialize with a specific prompt template.
        
        Args:
            template: The prompt template string
        """
        self.prompt_template = ChatPromptTemplate.from_template(template)
    
    def invoke(self, **kwargs) -> Dict:
        """
        Invoke the LLM with the prompt template and arguments.
        
        Args:
            **kwargs: Arguments to format the prompt template
            
        Returns:
            Parsed response from the LLM
        """
        try:
            # If LLM is not initialized, return an error
            if llm is None:
                logger.error("LLM is not initialized")
                return self.error_response("LLM is not initialized")
            
            # Format the prompt
            prompt_message = self.prompt_template.format(**kwargs)
            
            # Log formatted prompt
            logger.info(f"Invoking LLM with formatted prompt")
            
            # Invoke the LLM
            start_time = time.time()
            response = llm.invoke(prompt_message)
            elapsed = time.time() - start_time
            
            # Log the raw response before parsing
            content = response.content
            logger.info(f"LLM response received in {elapsed:.2f}s, length: {len(content)} chars")
            
            # Parse the response as JSON
            try:
                parsed = LLMUtils.parse_json_response(content)
                
                # Check if we got an error during parsing
                if "error" in parsed:
                    logger.warning(f"JSON parsing returned an error response: {parsed['error']}")
                    
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Raw response: {content}")
                return self.fallback_response()
            
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.error_response(str(e))
    
    def fallback_response(self) -> Dict:
        """
        Provide a fallback response when parsing fails.
        
        Returns:
            Fallback response dict
        """
        raise NotImplementedError("Subclasses must implement fallback_response()")
    
    def error_response(self, error_message: str) -> Dict:
        """
        Provide an error response when an exception occurs.
        
        Args:
            error_message: Error message from the exception
            
        Returns:
            Error response dict
        """
        raise NotImplementedError("Subclasses must implement error_response()")


class ActionDeterminationHandler(PromptHandler):
    """
    Handler for determining next actions.
    """
    
    def __init__(self):
        super().__init__(PromptTemplates.ACTION_DETERMINATION)
    
    def determine_action(self, user_details: Dict, active_context: Dict, 
                        current_workflow: Dict, available_workflows: List[Dict],
                        messages: List[Dict], action_types: List[str]) -> Dict:
        """
        Determine the next action based on current context.
        
        Args:
            user_details: Information about the user
            active_context: Active conversation context
            current_workflow: Current workflow being executed
            available_workflows: List of available workflows
            messages: Message history
            action_types: Available action types
            
        Returns:
            Action determination result
        """
        # Prepare context for the prompt
        user_entity_json = json.dumps(user_details.get('metadata', {}), indent=2)
        active_context_str = json.dumps(active_context, indent=2)
        workflow_name = current_workflow.get('name', 'No Workflow')
        workflow_steps = json.dumps(current_workflow.get('steps', {}), indent=2)
        
        # Format available workflows
        available_workflows_str = "\n".join([
            f"{workflow.get('name')}: {workflow.get('description', 'No description')}"
            for workflow in available_workflows
        ])
        
        # Format messages
        messages_str = LLMUtils.format_messages_history(messages)
        
        # Format action types
        action_types_str = "\n".join(action_types)
        
        return self.invoke(
            user_entity_json=user_entity_json,
            active_context=active_context_str,
            workflow_name=workflow_name,
            workflow_steps=workflow_steps,
            available_workflows=available_workflows_str,
            messages=messages_str,
            action_types=action_types_str
        )
    
    def fallback_response(self) -> Dict:
        return {
            "Thinking": "Failed to parse LLM response as JSON",
            "Uncertainty": "High - could not parse LLM response",
            "Step": "Error in processing",
            "Next Action": "Send Developer Message"
        }
    
    def error_response(self, error_message: str) -> Dict:
        return {
            "Thinking": f"Error invoking LLM: {error_message}",
            "Uncertainty": "High - LLM invocation failed",
            "Step": "Error in processing",
            "Next Action": "Send Developer Message"
        }


class MessageGenerationHandler(PromptHandler):
    """
    Handler for generating user messages.
    """
    
    def __init__(self):
        super().__init__(PromptTemplates.MESSAGE_GENERATION)
    
    def generate_message(self, user_details: Dict, active_context: Dict, 
                       current_workflow: Dict, messages: List[Dict]) -> Dict:
        """
        Generate a message to send to the user.
        
        Args:
            user_details: Information about the user
            active_context: Active conversation context
            current_workflow: Current workflow being executed
            messages: Message history
            
        Returns:
            Message generation result
        """
        user_entity_json = json.dumps(user_details.get('metadata', {}), indent=2)
        active_context_str = json.dumps(active_context, indent=2)
        workflow_name = current_workflow.get('name', 'No Workflow')
        workflow_steps = json.dumps(current_workflow.get('steps', {}), indent=2)
        messages_str = LLMUtils.format_messages_history(messages)
        
        return self.invoke(
            user_entity_json=user_entity_json,
            active_context=active_context_str,
            workflow_name=workflow_name,
            workflow_steps=workflow_steps,
            messages=messages_str
        )
    
    def fallback_response(self) -> Dict:
        return {
            "Thinking": "Failed to parse LLM response as JSON",
            "Message": "I'm having trouble processing your request. Could you please try again or rephrase your message?"
        }
    
    def error_response(self, error_message: str) -> Dict:
        return {
            "Thinking": f"Error invoking LLM: {error_message}",
            "Message": "I'm currently experiencing technical difficulties. Please try again in a moment."
        }


class KnowledgebaseUpsertHandler(PromptHandler):
    """
    Handler for determining database upserts.
    """
    
    def __init__(self):
        super().__init__(PromptTemplates.KNOWLEDGEBASE_UPSERT)
    
    def determine_upsert(self, user_details: Dict, active_context: Dict, 
                       current_workflow: Dict, messages: List[Dict]) -> Dict:
        """
        Determine what data to upsert to the knowledgebase.
        
        Args:
            user_details: Information about the user
            active_context: Active conversation context
            current_workflow: Current workflow being executed
            messages: Message history
            
        Returns:
            Upsert determination result
        """
        user_entity_json = json.dumps(user_details.get('metadata', {}), indent=2)
        active_context_str = json.dumps(active_context, indent=2)
        workflow_name = current_workflow.get('name', 'No Workflow')
        workflow_steps = json.dumps(current_workflow.get('steps', {}), indent=2)
        messages_str = LLMUtils.format_messages_history(messages)
        
        return self.invoke(
            user_entity_json=user_entity_json,
            active_context=active_context_str,
            workflow_name=workflow_name,
            workflow_steps=workflow_steps,
            messages=messages_str
        )
    
    def fallback_response(self) -> Dict:
        return {
            "reasoning": "Failed to parse LLM response as JSON",
            "upsertJSON": {}
        }
    
    def error_response(self, error_message: str) -> Dict:
        return {
            "reasoning": f"Error invoking LLM: {error_message}",
            "upsertJSON": {}
        }


class QueryGenerationHandler(PromptHandler):
    """
    Handler for generating database queries.
    """
    
    def __init__(self):
        super().__init__(PromptTemplates.QUERY_GENERATION)
    
    def generate_query(self, user_query: str, entity_types: List[str], db_schema: Dict) -> Dict:
        """
        Generate a database query from natural language.
        
        Args:
            user_query: Natural language query from user
            entity_types: Available entity types
            db_schema: Database schema information
            
        Returns:
            Query generation result
        """
        entity_types_str = ", ".join(entity_types)
        db_schema_str = json.dumps(db_schema, indent=2)
        
        return self.invoke(
            user_query=user_query,
            entity_types=entity_types_str,
            db_schema=db_schema_str
        )
    
    def fallback_response(self) -> Dict:
        return {
            "reasoning": "Failed to parse LLM response as JSON",
            "queryType": "fuzzy",
            "entityTypes": ["provider", "product", "service", "event", "note"],
            "filters": {},
            "searchText": "",
            "limit": 10
        }
    
    def error_response(self, error_message: str) -> Dict:
        return {
            "reasoning": f"Error invoking LLM: {error_message}",
            "queryType": "fuzzy",
            "entityTypes": ["provider", "product", "service", "event", "note"],
            "filters": {},
            "searchText": "",
            "limit": 10
        }


class ContextManagementHandler(PromptHandler):
    """
    Handler for managing conversation context.
    """
    
    def __init__(self):
        super().__init__(PromptTemplates.CONTEXT_MANAGEMENT)
    
    def update_context(self, current_context: Dict, messages: List[Dict], current_workflow: Dict) -> Dict:
        """
        Determine relevant context updates for the conversation.
        
        Args:
            current_context: Current active context
            messages: Recent message history
            current_workflow: Current workflow information
            
        Returns:
            Context update result
        """
        current_context_str = json.dumps(current_context, indent=2)
        messages_str = LLMUtils.format_messages_history(messages)
        current_workflow_str = json.dumps(current_workflow, indent=2)
        
        return self.invoke(
            current_context=current_context_str,
            messages=messages_str,
            current_workflow=current_workflow_str
        )
    
    def fallback_response(self) -> Dict:
        return {
            "reasoning": "Failed to parse LLM response as JSON",
            "keepContext": [],
            "removeContext": [],
            "addContext": {}
        }
    
    def error_response(self, error_message: str) -> Dict:
        return {
            "reasoning": f"Error invoking LLM: {error_message}",
            "keepContext": [],
            "removeContext": [],
            "addContext": {}
        }


class LLMHandler:
    """
    Main LLM handler that provides access to all prompt handlers.
    """
    
    _action_handler = None
    _message_handler = None
    _upsert_handler = None
    _query_handler = None
    _context_handler = None
    
    @property
    def action_handler(self) -> ActionDeterminationHandler:
        if not LLMHandler._action_handler:
            LLMHandler._action_handler = ActionDeterminationHandler()
        return LLMHandler._action_handler
    
    @property
    def message_handler(self) -> MessageGenerationHandler:
        if not LLMHandler._message_handler:
            LLMHandler._message_handler = MessageGenerationHandler()
        return LLMHandler._message_handler
    
    @property
    def upsert_handler(self) -> KnowledgebaseUpsertHandler:
        if not LLMHandler._upsert_handler:
            LLMHandler._upsert_handler = KnowledgebaseUpsertHandler()
        return LLMHandler._upsert_handler
    
    @property
    def query_handler(self) -> QueryGenerationHandler:
        if not LLMHandler._query_handler:
            LLMHandler._query_handler = QueryGenerationHandler()
        return LLMHandler._query_handler
    
    @property
    def context_handler(self) -> ContextManagementHandler:
        if not LLMHandler._context_handler:
            LLMHandler._context_handler = ContextManagementHandler()
        return LLMHandler._context_handler
    
    @staticmethod
    def determine_action(user_details: Dict, active_context: Dict, 
                       current_workflow: Dict, available_workflows: List[Dict],
                       messages: List[Dict], action_types: List[str]) -> Dict:
        """
        Determine the next action based on current context.
        """
        handler = LLMHandler().action_handler
        return handler.determine_action(
            user_details, active_context, current_workflow, 
            available_workflows, messages, action_types
        )
    
    @staticmethod
    def generate_user_message(user_details: Dict, active_context: Dict, 
                            current_workflow: Dict, messages: List[Dict]) -> Dict:
        """
        Generate a message to send to the user.
        """
        handler = LLMHandler().message_handler
        return handler.generate_message(
            user_details, active_context, current_workflow, messages
        )
    
    @staticmethod
    def determine_upsert(user_details: Dict, active_context: Dict, 
                       current_workflow: Dict, messages: List[Dict]) -> Dict:
        """
        Determine what data to upsert to the knowledgebase.
        """
        handler = LLMHandler().upsert_handler
        return handler.determine_upsert(
            user_details, active_context, current_workflow, messages
        )
    
    @staticmethod
    def generate_query(user_query: str, entity_types: List[str], db_schema: Dict) -> Dict:
        """
        Generate a database query from natural language.
        """
        handler = LLMHandler().query_handler
        return handler.generate_query(user_query, entity_types, db_schema)
    
    @staticmethod
    def update_context(current_context: Dict, messages: List[Dict], current_workflow: Dict) -> Dict:
        """
        Determine relevant context updates for the conversation.
        """
        handler = LLMHandler().context_handler
        return handler.update_context(current_context, messages, current_workflow)


# Add DEBUG mode test if this file is run directly
if __name__ == "__main__":
    print("LLM Module Test")
    print("-" * 40)
    print(f"Using model: {LLM_MODEL}")
    print(f"API Key available: {'Yes' if GROQ_API_KEY else 'No'}")
    print("-" * 40)
    
    if llm:
        try:
            # Test with a simple prompt
            print("Testing LLM with a simple prompt...")
            response = llm.invoke("Hello, please respond with valid JSON. Format: {\"status\": \"ok\"}")
            print(f"Response: {response.content}")
            
            # Test JSON parsing
            print("\nTesting JSON extraction...")
            extracted = LLMUtils.extract_json_from_response(response.content)
            print(f"Extracted: {extracted}")
            
            # Test action determination
            print("\nTesting action determination...")
            handler = ActionDeterminationHandler()
            result = handler.determine_action(
                user_details={"metadata": {}},
                active_context={},
                current_workflow={"name": "determine_workflow", "steps": {"1": "Analyze user message"}},
                available_workflows=[],
                messages=[{"direction": "inbound", "message_content": "Hello", "timestamp": "2023-01-01"}],
                action_types=["Send User Message", "Wait"]
            )
            print(f"Result: {json.dumps(result, indent=2)}")
            
            print("\nAll tests completed successfully!")
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            print(traceback.format_exc())
    else:
        print("LLM is not initialized. Cannot run tests.")