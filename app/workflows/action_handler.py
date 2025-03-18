"""
Action handler module for managing workflow execution.
"""

from typing import Dict, List, Optional, Any, Union
import json

from app.models.llm import LLMHandler
from app.database.supabase_client import SupabaseDB

# Define action types
ACTION_TYPES = [
    "Send User Message",
    "Wait",
    "Upsert to Knowledgebase",
    "Query Knowledgebase",
    "Reset User Active Context",
    "Prompt LLM",
    "Send Developer Message",
    "Add Log"
]

# Define default workflows for MVP 0.01
DEFAULT_WORKFLOWS = [
    {
        "name": "determine_workflow",
        "description": "Default entry point to determine the appropriate workflow based on user intent",
        "steps": {
            "1": "Analyze user message to understand intent",
            "2": "Determine if user is trying to find, add, or edit information",
            "3": "Select appropriate follow-up workflow",
            "4": "Inform user of next steps"
        }
    },
    {
        "name": "add_new_provider",
        "description": "Workflow to add a new provider to the directory",
        "steps": {
            "1": "Ask for provider name",
            "2": "Ask for provider type/category",
            "3": "Ask for contact information",
            "4": "Ask for location",
            "5": "Confirm information",
            "6": "Save provider to database",
            "7": "Confirm successful addition"
        }
    },
    {
        "name": "find_entity",
        "description": "Workflow to search for entities in the directory",
        "steps": {
            "1": "Ask what the user is looking for",
            "2": "Search for matching entities",
            "3": "Present results to user",
            "4": "Ask if user wants more details on a specific result",
            "5": "Provide detailed information if requested"
        }
    }
]

class ActionHandler:
    """
    Handles the execution of actions based on workflow state.
    """
    
    @staticmethod
    def process_incoming_message(phone_number: str, message_content: str) -> Dict:
        """
        Process an incoming message from a user.
        
        Args:
            phone_number: User's phone number
            message_content: Content of the message
            
        Returns:
            Response dict with message to send back
        """
        # Get or create user
        user = SupabaseDB.get_or_create_user(phone_number)
        if not user:
            # Log error and return generic message
            SupabaseDB.add_log(
                message=f"Failed to get or create user for {phone_number}",
                level='error'
            )
            return {
                "message": "Sorry, we're experiencing technical difficulties. Please try again later."
            }
            
        user_id = user['id']
        
        # Add the incoming message to the database
        message = SupabaseDB.add_message(
            user_id=user_id,
            content=message_content,
            direction='inbound'
        )
        
        # Get recent message history
        messages = SupabaseDB.get_user_messages(user_id)
        messages.reverse()  # Chronological order
        
        # Get or create active workflow
        # For MVP 0.01, we simplify by creating a new workflow instance for each message
        workflow = SupabaseDB.get_or_create_workflow(
            name="determine_workflow",
            user_id=user_id,
            description="Determine appropriate workflow",
            steps=DEFAULT_WORKFLOWS[0]["steps"]
        )
        
        if not workflow:
            # Log error and return generic message
            SupabaseDB.add_log(
                message=f"Failed to create workflow for user {user_id}",
                level='error',
                user_id=user_id
            )
            return {
                "message": "Sorry, we're experiencing technical difficulties. Please try again later."
            }
            
        workflow_id = workflow['id']
        
        # Update message with workflow ID
        if message:
            SupabaseDB.add_message(
                user_id=user_id,
                content=message_content,
                direction='inbound',
                workflow_id=workflow_id,
                metadata={"message_id": message['id']}
            )
        
        # Initialize context
        active_context = {}
        
        # Determine next action
        action_result = LLMHandler.determine_action(
            user_details=user,
            active_context=active_context,
            current_workflow=workflow,
            available_workflows=DEFAULT_WORKFLOWS,
            messages=messages,
            action_types=ACTION_TYPES
        )
        
        # Create action record
        action = SupabaseDB.create_action(
            workflow_id=workflow_id,
            action_type=action_result.get('Next Action', 'Send User Message'),
            description=action_result.get('Step', 'Processing user message'),
            thinking=action_result.get('Thinking', 'Determining next action')
        )
        
        if not action:
            # Log error and return generic message
            SupabaseDB.add_log(
                message=f"Failed to create action for workflow {workflow_id}",
                level='error',
                user_id=user_id
            )
            return {
                "message": "Sorry, we're experiencing technical difficulties. Please try again later."
            }
            
        # Execute the action
        return ActionHandler.execute_action(
            action_type=action_result.get('Next Action', 'Send User Message'),
            user=user,
            active_context=active_context,
            workflow=workflow,
            messages=messages,
            action_id=action['id']
        )
    
    @staticmethod
    def execute_action(action_type: str, user: Dict, active_context: Dict,
                     workflow: Dict, messages: List[Dict], action_id: str) -> Dict:
        """
        Execute a specific action.
        
        Args:
            action_type: Type of action to execute
            user: User information
            active_context: Active context for the conversation
            workflow: Current workflow
            messages: Message history
            action_id: ID of the action record
            
        Returns:
            Result of the action
        """
        user_id = user['id']
        
        # Execute based on action type
        if action_type == "Send User Message":
            # Generate message
            message_result = LLMHandler.generate_user_message(
                user_details=user,
                active_context=active_context,
                current_workflow=workflow,
                messages=messages
            )
            
            # Add message to database
            response_message = message_result.get('Message', 'Sorry, I couldn\'t generate a response.')
            SupabaseDB.add_message(
                user_id=user_id,
                content=response_message,
                direction='outbound',
                workflow_id=workflow['id']
            )
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "thinking": message_result.get('Thinking', ''),
                    "message": response_message
                }
            )
            
            return {
                "message": response_message
            }
            
        elif action_type == "Upsert to Knowledgebase":
            # Determine what to upsert
            upsert_result = LLMHandler.determine_upsert(
                user_details=user,
                active_context=active_context,
                current_workflow=workflow,
                messages=messages
            )
            
            # Process the upsert based on the result
            reasoning = upsert_result.get('reasoning', '')
            upsert_json = upsert_result.get('upsertJSON', {})
            
            # For MVP 0.01, we simplify by just logging the upsert data
            # In a more complete implementation, this would update entities based on the data
            SupabaseDB.add_log(
                message=f"Upsert to Knowledgebase requested",
                level='info',
                user_id=user_id,
                action_id=action_id,
                details={
                    "reasoning": reasoning,
                    "upsertJSON": upsert_json
                }
            )
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "reasoning": reasoning,
                    "upsertJSON": upsert_json
                }
            )
            
            # After upserting, we usually want to send a confirmation message
            # Let's recursively call execute_action for Send User Message
            return ActionHandler.execute_action(
                action_type="Send User Message",
                user=user,
                active_context=active_context,
                workflow=workflow,
                messages=messages,
                action_id=action_id
            )
            
        elif action_type == "Query Knowledgebase":
            # For MVP 0.01, we implement a simplified search
            # Extract search terms from the latest user message
            latest_message = messages[-1]['message_content'] if messages else ""
            entity_types = ["provider", "product", "service", "event", "note"]
            db_schema = {"entities": {"entity_type": "string", "entity_name": "string"}}
            
            # Generate query
            query_result = LLMHandler.generate_query(
                user_query=latest_message,
                entity_types=entity_types,
                db_schema=db_schema
            )
            
            # Perform search based on the query
            search_results = SupabaseDB.search_entities(
                entity_type=query_result.get('entityTypes', [])[0] if query_result.get('entityTypes') else None,
                search_term=query_result.get('searchText', '')
            )
            
            # Update active context with search results
            active_context['search_results'] = search_results
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "query": query_result,
                    "results_count": len(search_results)
                }
            )
            
            # After querying, we usually want to send the results as a message
            # Let's recursively call execute_action for Send User Message
            return ActionHandler.execute_action(
                action_type="Send User Message",
                user=user,
                active_context=active_context,
                workflow=workflow,
                messages=messages,
                action_id=action_id
            )
            
        elif action_type == "Reset User Active Context":
            # For MVP 0.01, we just clear the context and log it
            old_context = active_context.copy()
            active_context.clear()
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "old_context": old_context
                }
            )
            
            # After resetting context, we usually want to confirm or ask a new question
            # Let's recursively call execute_action for Send User Message
            return ActionHandler.execute_action(
                action_type="Send User Message",
                user=user,
                active_context=active_context,
                workflow=workflow,
                messages=messages,
                action_id=action_id
            )
            
        elif action_type == "Wait":
            # For MVP 0.01, "Wait" is essentially a no-op since we're processing one message at a time
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "waiting_for": "user response"
                }
            )
            
            # In this case, we don't send a message back
            return {
                "message": None
            }
            
        elif action_type == "Send Developer Message":
            # For MVP 0.01, we log the developer message
            SupabaseDB.add_log(
                message="Developer notification triggered",
                level='warn',
                user_id=user_id,
                action_id=action_id,
                details={
                    "workflow_id": workflow['id'],
                    "latest_message": messages[-1]['message_content'] if messages else ""
                }
            )
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete'
            )
            
            # Send a message to the user
            return {
                "message": "I'm experiencing some difficulties processing your request. Our developers have been notified. In the meantime, could you try again with a different request?"
            }
            
        elif action_type == "Add Log":
            # For MVP 0.01, just create a log entry
            SupabaseDB.add_log(
                message="System log from action handler",
                level='info',
                user_id=user_id,
                action_id=action_id,
                details={
                    "workflow_id": workflow['id'],
                    "context": active_context
                }
            )
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete'
            )
            
            # Usually after logging, we'd do something else
            return ActionHandler.execute_action(
                action_type="Send User Message",
                user=user,
                active_context=active_context,
                workflow=workflow,
                messages=messages,
                action_id=action_id
            )
            
        else:
            # Unrecognized action type
            SupabaseDB.add_log(
                message=f"Unrecognized action type: {action_type}",
                level='error',
                user_id=user_id,
                action_id=action_id
            )
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Error',
                execution_data={
                    "error": f"Unrecognized action type: {action_type}"
                }
            )
            
            # Send a generic message
            return {
                "message": "I'm not sure how to handle that request right now. Could you try something else?"
            }