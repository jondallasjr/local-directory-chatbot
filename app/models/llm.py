"""
LLM integration module using LangChain.

This module provides a modular approach to LLM interactions with specialized
prompt handlers for different types of operations in the Directory Chatbot.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from app.config.settings import GROQ_API_KEY, LLM_MODEL

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=LLM_MODEL
)

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

Respond in perfect JSON. Start your response with and open bracket, and end it with a closign bracket.
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

Respond in perfectly formatted JSON with keys 'Thinking' and 'Message'. Start your response with { and end it with }
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
        if not content.strip().startswith('{') or not content.strip().endswith('}'):
            # Try to extract JSON from the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                return content[start_idx:end_idx]
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
        # Try to extract JSON from the response
        try:
            # First try to parse as is
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the content
            try:
                content = content.strip()
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    extracted_json = content[start_idx:end_idx]
                    return json.loads(extracted_json)
                else:
                    logger.error(f"Could not extract JSON from content: {content}")
                    raise ValueError("No JSON content found in response")
            except Exception as e:
                logger.error(f"JSON extraction error: {e}")
                logger.error(f"Content: {content}")
                raise
        
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
            prompt_message = self.prompt_template.format(**kwargs)
            logger.debug(f"Sending prompt to LLM: {prompt_message}")
            
            response = llm.invoke(prompt_message)
            content = response.content
            logger.debug(f"Raw LLM response: {content}")
            
            parsed = LLMUtils.parse_json_response(content)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Raw response: {content}")
            return self.fallback_response()
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
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
        if not self._action_handler:
            self._action_handler = ActionDeterminationHandler()
        return self._action_handler
    
    @property
    def message_handler(self) -> MessageGenerationHandler:
        if not self._message_handler:
            self._message_handler = MessageGenerationHandler()
        return self._message_handler
    
    @property
    def upsert_handler(self) -> KnowledgebaseUpsertHandler:
        if not self._upsert_handler:
            self._upsert_handler = KnowledgebaseUpsertHandler()
        return self._upsert_handler
    
    @property
    def query_handler(self) -> QueryGenerationHandler:
        if not self._query_handler:
            self._query_handler = QueryGenerationHandler()
        return self._query_handler
    
    @property
    def context_handler(self) -> ContextManagementHandler:
        if not self._context_handler:
            self._context_handler = ContextManagementHandler()
        return self._context_handler
    
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