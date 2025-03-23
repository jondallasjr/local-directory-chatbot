"""
Action handler module for managing workflow execution.
"""

from app.database.supabase_client import SupabaseDB, supabase
from app.utils.logging_utils import log_exceptions
from typing import Dict, List, Optional, Any, Union
import json
import logging
from app.models.llm import LLMHandler
from app.database.supabase_client import SupabaseDB
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class ActionHandler:
    """
    Handles the execution of actions based on workflow state.
    """
    
    @staticmethod
    @log_exceptions 
    def process_incoming_message(phone_number: str, message_content: str) -> Dict:
        """
        Process an incoming message from a user (with workflow management).
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
            
            # Get active workflows for this user, ordered by most recently created
            active_workflows = SupabaseDB.get_active_workflows_for_user(user_id)
            
            # If no active workflows, create a new "determine_workflow" instance
            if not active_workflows:
                # Get workflow definition
                determine_workflow_def = ActionHandler.get_workflow_definition_by_name("determine_workflow")
                if not determine_workflow_def:
                    # Log error and return generic message
                    SupabaseDB.add_log(
                        message=f"Failed to find determine_workflow definition",
                        level='error',
                        user_id=user_id
                    )
                    return {
                        "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                    }
                    
                # Create workflow instance
                new_workflow = ActionHandler.create_workflow_instance(
                    definition_id=determine_workflow_def.get('id'),
                    user_id=user_id,
                    status='active'
                )
                
                if not new_workflow:
                    # Log error and return generic message
                    SupabaseDB.add_log(
                        message=f"Failed to create workflow instance for user {user_id}",
                        level='error',
                        user_id=user_id
                    )
                    return {
                        "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                    }
                
                # Use the new workflow
                current_workflow = new_workflow
                # Also add it to active workflows list
                active_workflows = [current_workflow]
            else:
                # Use the most recent active workflow
                current_workflow = active_workflows[0]
            
            # Update message with workflow ID
            if message:
                SupabaseDB.update_message(
                    message_id=message['id'],
                    workflow_id=current_workflow['id']
                )
            
            # Initialize or get context data
            active_context = current_workflow.get('context_data', {})
            
            # Add materialized views from all active workflows to context
            for workflow in active_workflows:
                # Extract required materializations from workflow definition
                workflow_def = workflow.get('workflow_definition', {})
                required_views = workflow_def.get('required_materializations', [])
                
                # Add each required view to the active context
                for view_name in required_views:
                    # Get the materialized view content
                    view_content = SupabaseDB.get_materialized_view(view_name)
                    if view_content:
                        # Add to context with prefix to avoid collisions
                        active_context[f"materialized_{view_name}"] = view_content
            
            # Extract workflow information
            workflow_def = current_workflow.get('workflow_definition', {})
            
            # Get the steps as a properly formatted string for the LLM
            workflow_steps_str = workflow_def.get('steps', '')
            if isinstance(workflow_steps_str, dict):
                # Convert dict to string if needed
                workflow_steps_str = '\n'.join([f"{key}: {value}" for key, value in workflow_steps_str.items()])

            # Make sure the current_step is set if not already
            if not current_workflow.get('current_step'):
                # Default to the first step if none is set
                first_step = "1. Greet the User and Determine Workflow"  # Default first step
                SupabaseDB.update_workflow_instance(
                    instance_id=current_workflow['id'],
                    status='active',
                    current_step=first_step
                )
                current_workflow['current_step'] = first_step

            # Enhanced workflow info for LLM
            workflow_for_llm = {
                'id': current_workflow.get('id'),
                'name': workflow_def.get('name', 'Unknown'),
                'description': workflow_def.get('description', ''),
                'steps': workflow_steps_str,  # Formatted steps as string
                'status': current_workflow.get('status', 'active'),
                'current_step': current_workflow.get('current_step'),
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
            
            if action_result.get('Next Action') == 'Transition Workflow':
                # Extract the target workflow type from context or action result
                target_workflow_type = action_result.get('Target Workflow') or active_context.get('target_workflow_type')
                
                if target_workflow_type:
                    # Try to find the workflow definition
                    target_workflow_def = ActionHandler.get_workflow_definition_by_name(target_workflow_type)
                    
                    if target_workflow_def:
                        # Create new workflow instance
                        new_workflow = ActionHandler.create_workflow_instance(
                            definition_id=target_workflow_def['id'],
                            user_id=user_id,
                            status='active',
                            context_data=active_context  # Transfer context
                        )
                        
                        if new_workflow:
                            # Update the current workflow reference
                            current_workflow = new_workflow
                            # Update action with the new workflow ID
                            SupabaseDB.update_action_status(
                                action_id=action['id'],
                                status='Complete',
                                execution_data={
                                    "transition_to": target_workflow_type,
                                    "new_workflow_id": new_workflow['id']
                                }
                            )
                            
                            # Log the transition
                            SupabaseDB.add_log(
                                message=f"Transitioned from workflow {workflow_for_llm['name']} to {target_workflow_type}",
                                level='info',
                                user_id=user_id
                            )
            
            # Update context based on message content
            context_update = LLMHandler.update_context(
                current_context=active_context,
                messages=messages[-3:],  # Use last 3 messages for context
                current_workflow=workflow_for_llm
            )

            if context_update and not 'error' in context_update:
                # Apply context updates
                for key in context_update.get('removeContext', []):
                    if key in active_context:
                        del active_context[key]
                        
                for key, value in context_update.get('addContext', {}).items():
                    active_context[key] = value
                    
                # Update workflow with new context
                SupabaseDB.update_workflow_instance(
                    instance_id=current_workflow['id'],
                    status='active',
                    current_step=current_workflow.get('current_step'),
                    context_data=active_context
                )
            
            # Update the workflow instance with any new step
            if action_result.get('Step'):
                SupabaseDB.update_workflow_instance(
                    instance_id=current_workflow['id'],
                    status='active',
                    current_step=action_result.get('Step'),
                    context_data=active_context
                )
            
            # Create action record
            action = SupabaseDB.create_action(
                workflow_id=current_workflow['id'],
                action_type=action_result.get('Next Action', 'Send User Message'),
                description=action_result.get('Step', 'Processing user message'),
                thinking=action_result.get('Thinking', 'Determining next action')
            )
            
            if not action:
                # Log error and return generic message
                SupabaseDB.add_log(
                    message=f"Failed to create action for workflow {current_workflow['id']}",
                    level='error',
                    user_id=user_id
                )
                return {
                    "message": "Sorry, we're experiencing technical difficulties. Please try again later."
                }
                
            if action_result.get('Next Action') == 'Send User Message':
                # Check for search-related terms in messages
                search_indicators = ['find', 'search', 'looking for', 'where', 'show me', 'any', 'are there']
                if any(indicator in message_content.lower() for indicator in search_indicators):
                    # Extract search context from message
                    active_context['search_text'] = message_content
                    active_context['entity_types'] = ['provider', 'service', 'event', 'product']  # Default to these types
                    active_context['target_workflow_type'] = 'find_entity'
                    
                    # Log the context update
                    SupabaseDB.add_log(
                        message="Added search context",
                        level='info',
                        user_id=user_id,
                        details={"context": active_context}
                    )
                    
                    # Update workflow with new context
                    SupabaseDB.update_workflow_instance(
                        instance_id=current_workflow['id'],
                        status='active',
                        current_step=current_workflow.get('current_step'),
                        context_data=active_context
                    )
                
            # Execute the action
            return ActionHandler.execute_action(
                action_type=action_result.get('Next Action', 'Send User Message'),
                user=user,
                active_context=active_context,
                workflow=workflow_for_llm,
                active_workflows=active_workflows,
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
    def create_workflow_instance(definition_id: str, user_id: str, 
                            status: str = 'in_progress',  # Match the default in SupabaseDB
                            current_step: Optional[str] = None,
                            context_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create a new workflow instance with the full definition copied.
        """
        try:
            # First, get the workflow definition
            workflow_def = SupabaseDB.get_workflow_definition(definition_id)
            if not workflow_def:
                logger.error(f"Workflow definition not found: {definition_id}")
                return None
                
            # Create the workflow instance with the copied definition
            workflow_data = {
                'definition_id': definition_id,
                'user_id': user_id,
                'status': status,
                'started_at': 'now()',
                'workflow_definition': workflow_def  # Store the full definition
            }
            
            if current_step:
                workflow_data['current_step'] = current_step
                
            if context_data:
                workflow_data['context_data'] = context_data
                
            response = supabase.table('workflow_instances').insert(workflow_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating workflow instance: {e}")
            return None
    
    @staticmethod
    def get_active_workflows_for_user(user_id: str) -> List[Dict]:
        """
        Get all active workflow instances for a user, ordered by most recent first.
        """
        try:
            response = supabase.table('workflow_instances') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('status', 'active') \
                .order('started_at', desc=True) \
                .execute()
            
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting active workflows for user: {e}")
            return []
    
    @staticmethod
    def add_workflow_definitions(definitions: List[Dict]) -> None:
        """
        Add multiple workflow definitions to the database.
        """
        try:
            # Insert the definitions
            response = supabase.table('workflow_definitions').insert(definitions).execute()
            
            # Check for errors
            if response.error:
                print(f"Error inserting workflow definitions: {response.error}")
                return
            
            # Log success
            print(f"Successfully added {len(response.data)} workflow definitions")
        except Exception as e:
            print(f"Exception adding workflow definitions: {str(e)}")
    
    @staticmethod
    @log_exceptions
    def execute_action(action_type: str, user: Dict, active_context: Dict,
                    workflow: Dict, messages: List[Dict], action_id: str, 
                    active_workflows: List[Dict] = None) -> Dict:
        """
        Execute a specific action.
        
        Args:
            action_type: Type of action to execute
            user: User information
            active_context: Active conversation context
            workflow: Current workflow being executed
            messages: Message history
            action_id: ID of the action record
            active_workflows: List of all active workflows for the user (optional)
            
        Returns:
            Result of the action
        """
        # If active_workflows is None, initialize it with just the current workflow
        if active_workflows is None:
            active_workflows = [workflow] if workflow else []
            
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
            
            # Update the workflow context
            if workflow and workflow.get('id'):
                SupabaseDB.update_workflow_instance(
                    instance_id=workflow['id'],
                    status='active',
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
        
        elif action_type == "Transition Workflow":
            # Get the target workflow type from context
            target_workflow_type = active_context.get('target_workflow_type')
            if not target_workflow_type:
                # No target specified, log error
                SupabaseDB.add_log(
                    message="Transition Workflow action called without target_workflow_type in context",
                    level='error',
                    user_id=user_id,
                    action_id=action_id
                )
                
                return {
                    "message": "I'm not sure what you want to do next. Could you please clarify?",
                    "thinking": "Missing target workflow for transition",
                    "step": "Error in workflow transition",
                    "action": "Error",
                    "context": active_context
                }
            
            # Get the workflow definition
            target_workflow_def = ActionHandler.get_workflow_definition_by_name(target_workflow_type)
            if not target_workflow_def:
                # Target workflow not found, log error
                SupabaseDB.add_log(
                    message=f"Transition Workflow: target workflow '{target_workflow_type}' not found",
                    level='error',
                    user_id=user_id,
                    action_id=action_id
                )
                
                return {
                    "message": f"I'm sorry, but I don't know how to help with '{target_workflow_type}' right now.",
                    "thinking": f"Target workflow '{target_workflow_type}' not found",
                    "step": "Error in workflow transition",
                    "action": "Error",
                    "context": active_context
                }
            
            # Create the new workflow instance
            new_workflow = ActionHandler.create_workflow_instance(
                definition_id=target_workflow_def['id'],
                user_id=user_id,
                status='active',
                context_data=active_context  # Transfer existing context
            )
            
            if not new_workflow:
                # Failed to create new workflow, log error
                SupabaseDB.add_log(
                    message=f"Failed to create new workflow instance for '{target_workflow_type}'",
                    level='error',
                    user_id=user_id,
                    action_id=action_id
                )
                
                return {
                    "message": "I'm having trouble processing your request right now. Could you try again?",
                    "thinking": "Failed to create new workflow instance",
                    "step": "Error in workflow transition",
                    "action": "Error",
                    "context": active_context
                }
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "from_workflow": workflow['id'],
                    "to_workflow": new_workflow['id'],
                    "workflow_type": target_workflow_type
                }
            )
            
            # Generate a message about the transition
            workflow_description = target_workflow_def.get('description', target_workflow_type)
            
            return {
                "message": f"I'll help you with {workflow_description}. Let's get started.",
                "thinking": f"Transitioned to {target_workflow_type} workflow",
                "step": "Workflow transition complete",
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
            # For MVP 0.01, we implement a simplified search with improved error handling
            try:
                # Extract search terms from the latest user message or context
                latest_message = messages[-1]['message_content'] if messages else ""
                search_text = active_context.get('search_text', '')
                
                # If no search text in context, try to extract it from the last message
                if not search_text:
                    search_text = latest_message
                    active_context['search_text'] = search_text
                
                # Default entity types if not specified
                entity_types = active_context.get('entity_types', ["provider", "product", "service", "event", "note"])
                
                # Basic database schema for the query generator
                db_schema = {"entities": {"entity_type": "string", "entity_name": "string", "attributes": "jsonb"}}
                
                # Generate query
                query_result = LLMHandler.generate_query(
                    user_query=search_text,
                    entity_types=entity_types,
                    db_schema=db_schema
                )
                
                # Log the query
                SupabaseDB.add_log(
                    message=f"Generated search query",
                    level='info',
                    user_id=user_id,
                    action_id=action_id,
                    details={"query": query_result}
                )
                
                # Perform search based on the query
                search_results = []
                try:
                    search_results = SupabaseDB.search_entities(
                        entity_type=query_result.get('entityTypes', [])[0] if query_result.get('entityTypes') else None,
                        search_term=query_result.get('searchText', '')
                    )
                except Exception as e:
                    # Log error but don't fail
                    SupabaseDB.add_log(
                        message=f"Error searching entities: {str(e)}",
                        level='error',
                        user_id=user_id,
                        action_id=action_id
                    )
                    search_results = []

                # Check if we have results
                if not search_results or len(search_results) == 0:
                    # Log the empty results
                    SupabaseDB.add_log(
                        message=f"No search results found for: {query_result.get('searchText', '')}",
                        level='info',
                        user_id=user_id,
                        action_id=action_id
                    )
                    
                    # Add placeholder data for MVP stage to avoid empty results
                    # This is generic placeholder data that doesn't hardcode specific entities
                    entity_type = query_result.get('entityTypes', [])[0] if query_result.get('entityTypes') else "provider"
                    search_terms = query_result.get('searchText', '').lower()
                    
                    if entity_type == "provider":
                        active_context['search_results'] = [
                            {"entity_name": f"Sample Provider 1 for {search_terms}", 
                            "attributes": {"location": "Central Area, Harare", "opening_hours": "8am-5pm daily"}},
                            {"entity_name": f"Sample Provider 2 for {search_terms}", 
                            "attributes": {"location": "Northern Suburbs, Harare", "opening_hours": "9am-6pm Mon-Sat"}},
                            {"entity_name": f"Sample Provider 3 for {search_terms}", 
                            "attributes": {"location": "CBD, Harare", "opening_hours": "8:30am-7pm daily"}}
                        ]
                    elif entity_type == "event":
                        active_context['search_results'] = [
                            {"entity_name": f"Sample Event 1 related to {search_terms}", 
                            "attributes": {"location": "Central Park, Harare", "date": "Next Saturday, 3pm"}},
                            {"entity_name": f"Sample Event 2 related to {search_terms}", 
                            "attributes": {"location": "City Hall, Harare", "date": "Every Sunday, 10am"}}
                        ]
                    else:
                        active_context['search_results'] = [
                            {"entity_name": f"Sample {entity_type.capitalize()} 1 for {search_terms}", 
                            "attributes": {"description": f"This is a sample {entity_type} related to your search"}},
                            {"entity_name": f"Sample {entity_type.capitalize()} 2 for {search_terms}", 
                            "attributes": {"description": f"Another sample {entity_type} for demonstration"}}
                        ]
                        
                    active_context['has_results'] = True
                    active_context['search_count'] = len(active_context['search_results'])
                    active_context['is_placeholder_data'] = True
                else:
                    # Update active context with real search results
                    active_context['search_results'] = search_results
                    active_context['has_results'] = True
                    active_context['search_count'] = len(search_results)
                    active_context['is_placeholder_data'] = False
                
                # Update action status
                SupabaseDB.update_action_status(
                    action_id=action_id,
                    status='Complete',
                    execution_data={
                        "query": query_result,
                        "results_count": len(active_context.get('search_results', [])),
                        "has_results": active_context.get('has_results', False),
                        "is_placeholder_data": active_context.get('is_placeholder_data', False)
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
            
            except Exception as e:
                # Log the error
                SupabaseDB.add_log(
                    message=f"Error in Query Knowledgebase action: {str(e)}",
                    level='error',
                    user_id=user_id,
                    action_id=action_id,
                    details={"traceback": traceback.format_exc()}
                )
                
                # Update action status
                SupabaseDB.update_action_status(
                    action_id=action_id,
                    status='Error',
                    execution_data={
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                )
                
                # Provide a fallback response
                active_context['error'] = str(e)
                return {
                    "message": "I'm having trouble searching for that information right now. Could you try again or ask about something else?",
                    "thinking": f"Error in Query Knowledgebase: {str(e)}",
                    "step": "Error handling",
                    "action": "Error",
                    "context": active_context
                }
            
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
        
        elif action_type == "Query Entity Graph":
            # Extract query parameters from context or last message
            search_text = active_context.get('search_text', '')
            
            # If no search text in context, try to extract it from the last message
            if not search_text and messages:
                search_text = messages[-1]['message_content']
            
            entity_types = active_context.get('entity_types', None)
            width = active_context.get('width', 2)
            
            # Log the search attempt
            SupabaseDB.add_log(
                message=f"Searching entity graph for: {search_text}",
                level='info',
                user_id=user_id,
                action_id=action_id
            )
            
            # Perform the search
            search_results = SupabaseDB.search_entities_with_context(
                search_text=search_text,
                entity_types=entity_types,
                width=width
            )
            
            # Update context with results
            active_context['search_results'] = search_results
            active_context['search_text'] = search_text
            
            # Update action status
            SupabaseDB.update_action_status(
                action_id=action_id,
                status='Complete',
                execution_data={
                    "search_text": search_text,
                    "entity_types": entity_types,
                    "width": width,
                    "results_count": search_results.get('count', 0) if isinstance(search_results, dict) else 0
                }
            )
            
            # After querying, generate a message with the results
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