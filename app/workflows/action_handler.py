"""
Action handler module for managing workflow execution.
"""

from typing import Dict, List, Optional, Any, Union
import json
import logging

from app.models.llm import LLMHandler
from app.database.supabase_client import SupabaseDB

# Configure logging
logger = logging.getLogger(__name__)

class ActionHandler:
    """
    Handles the execution of actions based on workflow state.
    """
    
    @staticmethod
    def get_action_types() -> List[str]:
        """
        Get available action types from the database.
        
        Returns:
            List of action type names
        Raises:
            Exception: If database query fails
        """
        action_types = SupabaseDB.get_action_types()
        return [action_type.get('name') for action_type in action_types if action_type.get('name')]
    
    @staticmethod
    def get_workflow_definitions() -> List[Dict]:
        """
        Get available workflow definitions from the database.
        
        Returns:
            List of workflow definitions
        Raises:
            Exception: If database query fails
        """
        return SupabaseDB.get_workflow_definitions()
    
    @staticmethod
    def get_workflow_definition_by_name(name: str) -> Optional[Dict]:
        """
        Get a workflow definition by name.
        
        Args:
            name: Name of the workflow to get
            
        Returns:
            Workflow definition or None if not found
        """
        try:
            response = SupabaseDB.supabase.table('workflow_definitions').select('*').eq('name', name).eq('active', True).limit(1).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            logger.error(f"No active workflow definition found with name: {name}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving workflow definition by name: {str(e)}")
            raise
    
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
        try:
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
            
            # See if the user has an active workflow already
            active_workflow = None
            try:
                # Check for any in-progress workflows for this user
                response = SupabaseDB.supabase.table('workflow_instances') \
                    .select('*') \
                    .eq('user_id', user_id) \
                    .eq('status', 'in_progress') \
                    .order('started_at', desc=True) \
                    .limit(1) \
                    .execute()
                
                if response.data and len(response.data) > 0:
                    active_workflow = response.data[0]
                    logger.info(f"Found active workflow for user: {active_workflow['id']}")
            except Exception as e:
                logger.warning(f"Error checking for active workflows: {e}")
            
            # If no active workflow, create a new one with the default workflow
            if not active_workflow:
                # Get workflow definition
                default_workflow_def = ActionHandler.get_workflow_definition_by_name("determine_workflow")
                if not default_workflow_def:
                    # Log error and return generic message
                    SupabaseDB.add_log(
                        message=f"Failed to find default workflow definition",
                        level='error',
                        user_id=user_id
                    )
                    return {
                        "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                    }
                    
                logger.info(f"Using workflow definition: {default_workflow_def['name']} (ID: {default_workflow_def['id']})")
                
                # Create workflow instance
                workflow_instance = SupabaseDB.create_workflow_instance(
                    definition_id=default_workflow_def.get('id'),
                    user_id=user_id,
                    status='in_progress'  # Mark as in_progress immediately
                )
        
                if not workflow_instance:
                    # Log error and return generic message
                    SupabaseDB.add_log(
                        message=f"Failed to create workflow instance for user {user_id}",
                        level='error',
                        user_id=user_id
                    )
                    return {
                        "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                    }
                
                workflow_id = workflow_instance['id']
                workflow_def = default_workflow_def
            else:
                # Use the active workflow
                workflow_id = active_workflow['id']
                
                # Get the workflow definition for this active workflow instance
                workflow_def = SupabaseDB.get_workflow_definition(active_workflow['definition_id'])
                if not workflow_def:
                    SupabaseDB.add_log(
                        message=f"Failed to find workflow definition for instance {workflow_id}",
                        level='error',
                        user_id=user_id
                    )
                    return {
                        "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                    }
                
                # Use the existing instance
                workflow_instance = active_workflow
            
            # Update message with workflow ID
            if message:
                SupabaseDB.add_message(
                    user_id=user_id,
                    content=message_content,
                    direction='inbound',
                    workflow_id=workflow_id,
                    metadata={"message_id": message['id']}
                )
            
            # Initialize context or load from existing workflow
            active_context = workflow_instance.get('context_data', {})
            
            # Get full workflow definition details
            workflow_for_llm = {
                'id': workflow_id,
                'instance_id': workflow_id,
                'definition_id': workflow_def.get('id'),
                'name': workflow_def.get('name'),
                'description': workflow_def.get('description'),
                'steps': workflow_def.get('steps', {}),
                'status': workflow_instance.get('status', 'in_progress'),
                'current_step': workflow_instance.get('current_step'),
                'context_data': active_context
            }
        
            # Determine next action
            action_result = LLMHandler.determine_action(
                user_details=user,
                active_context=active_context,
                current_workflow=workflow_for_llm,
                available_workflows=ActionHandler.get_workflow_definitions(),
                messages=messages,
                action_types=ActionHandler.get_action_types()
            )
            
            # Update the workflow instance with any new step
            if action_result.get('Step'):
                SupabaseDB.update_workflow_instance(
                    instance_id=workflow_id,
                    status='in_progress',
                    current_step=action_result.get('Step'),
                    context_data=active_context
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
                workflow=workflow_for_llm,
                messages=messages,
                action_id=action['id']
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            SupabaseDB.add_log(
                message=f"Error processing message: {str(e)}",
                level='error',
                user_id=user_id if locals().get('user_id') else None,
                details={"error": str(e), "message": message_content}
            )
            return {
                "message": "Sorry, I encountered an error while processing your message. Please try again later.",
                "error": str(e)
            }
    
    @staticmethod
    def execute_action(action_type: str, user: Dict, active_context: Dict,
                     workflow: Dict, messages: List[Dict], action_id: str) -> Dict:
        """
        Execute a specific action.
        
        Args:
            action_type: Type of action to execute
            user: User information
            active_context: Active conversation context
            workflow: Current workflow being executed
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
            
            # Also update the workflow context with any context changes
            if workflow and workflow.get('instance_id'):
                SupabaseDB.update_workflow_instance(
                    instance_id=workflow['instance_id'],
                    status='in_progress',
                    current_step=workflow.get('current_step'),
                    context_data=active_context
                )
            
            return {
                "message": response_message,
                "thinking": message_result.get('Thinking', ''),
                "step": workflow.get('current_step', 'Processing message'),
                "action": action_type,
                "context": active_context
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
                "message": None,
                "thinking": "Waiting for user response",
                "step": workflow.get('current_step', 'Waiting'),
                "action": action_type,
                "context": active_context
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
                "message": "I'm experiencing some difficulties processing your request. Our developers have been notified. In the meantime, could you try again with a different request?",
                "thinking": "Error encountered, notifying developers",
                "step": "Error handling",
                "action": action_type,
                "context": active_context
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
                "message": "I'm not sure how to handle that request right now. Could you try something else?",
                "thinking": f"Unrecognized action type: {action_type}",
                "step": "Error handling",
                "action": "Error",
                "context": active_context
            }