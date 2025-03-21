"""
Supabase client module for database operations.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
import uuid

from supabase import create_client, Client

from app.config.settings import SUPABASE_URL, SUPABASE_KEY

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class SupabaseDB:
    """
    Supabase database operations wrapper.
    Provides methods for CRUD operations on different tables.
    """
    # Add this line to make the client accessible as a class attribute
    supabase = supabase
    
    @staticmethod
    def get_user_by_phone(phone_number: str) -> Optional[Dict]:
        """
        Get a user by phone number or create if doesn't exist.
        """
        try:
            response = supabase.table('users').select('*').eq('phone_number', phone_number).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    @staticmethod
    def create_user(phone_number: str, role: str = 'user') -> Optional[Dict]:
        """
        Create a new user.
        """
        try:
            user_data = {
                'phone_number': phone_number,
                'role': role,
                'metadata': {}
            }
            response = supabase.table('users').insert(user_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    @staticmethod
    def get_or_create_user(phone_number: str) -> Optional[Dict]:
        """
        Get a user by phone number or create if doesn't exist.
        """
        user = SupabaseDB.get_user_by_phone(phone_number)
        if not user:
            user = SupabaseDB.create_user(phone_number)
        return user
    
    @staticmethod
    def add_message(user_id: str, content: str, direction: str, 
                   workflow_id: Optional[str] = None, 
                   entity_id: Optional[str] = None,
                   metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
        Add a new message to the database.
        """
        try:
            message_data = {
                'user_id': user_id,
                'message_content': content,
                'direction': direction,
                'metadata': metadata or {}
            }
            
            if workflow_id:
                message_data['related_workflow_id'] = workflow_id
            
            if entity_id:
                message_data['related_entity'] = entity_id
                
            response = supabase.table('messages').insert(message_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return None
    
    @staticmethod
    def get_user_messages(user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get recent messages for a user.
        """
        try:
            response = supabase.table('messages') \
                              .select('*') \
                              .eq('user_id', user_id) \
                              .order('timestamp', desc=True) \
                              .limit(limit) \
                              .execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    @staticmethod
    def create_or_update_entity(entity_type: str, entity_name: str, 
                              created_by: str, attributes: Dict,
                              entity_id: Optional[str] = None,
                              verification_status: str = 'unverified',
                              metadata: Optional[Dict] = None,
                              tags: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Create a new entity or update an existing one.
        """
        try:
            entity_data = {
                'entity_type': entity_type,
                'entity_name': entity_name,
                'created_by': created_by,
                'attributes': attributes,
                'verification_status': verification_status,
                'metadata': metadata or {},
                'tags': tags or []
            }
            
            if entity_id:
                entity_data['id'] = entity_id
                response = supabase.table('entities').update(entity_data).eq('id', entity_id).execute()
            else:
                response = supabase.table('entities').insert(entity_data).execute()
                
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating/updating entity: {e}")
            return None
    
    @staticmethod
    def search_entities(entity_type: Optional[str] = None, 
                      search_term: Optional[str] = None,
                      limit: int = 10) -> List[Dict]:
        """
        Search for entities by type and/or name.
        """
        try:
            query = supabase.table('entities').select('*')
            
            if entity_type:
                query = query.eq('entity_type', entity_type)
                
            if search_term:
                query = query.ilike('entity_name', f'%{search_term}%')
                
            response = query.limit(limit).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
    
    @staticmethod
    def get_workflow_definitions() -> List[Dict]:
        """
        Get all workflow definitions.
        """
        try:
            response = supabase.table('workflow_definitions').select('*').eq('active', True).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting workflow definitions: {e}")
            return []
    
    @staticmethod
    def get_workflow_definition(definition_id: str) -> Optional[Dict]:
        """
        Get a specific workflow definition.
        """
        try:
            response = supabase.table('workflow_definitions').select('*').eq('id', definition_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting workflow definition: {e}")
            return None
    
    @staticmethod
    def create_workflow_instance(definition_id: str, user_id: str, 
                               status: str = 'new',
                               current_step: Optional[str] = None,
                               context_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create a new workflow instance.
        """
        try:
            workflow_data = {
                'definition_id': definition_id,
                'user_id': user_id,
                'status': status,
                'started_at': 'now()'
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
    def update_workflow_instance(instance_id: str, status: str, 
                               current_step: Optional[str] = None,
                               context_data: Optional[Dict] = None) -> bool:
        """
        Update a workflow instance.
        """
        try:
            update_data = {'status': status}
            if current_step:
                update_data['current_step'] = current_step
            if context_data:
                update_data['context_data'] = context_data
                
            response = supabase.table('workflow_instances').update(update_data).eq('id', instance_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating workflow instance: {e}")
            return False
    
    @staticmethod
    def get_action_types() -> List[Dict]:
        """
        Get all action types.
        """
        try:
            response = supabase.table('action_types').select('*').execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting action types: {e}")
            return []
    
    @staticmethod
    def create_action(workflow_id: str, action_type: str, 
                    description: str, thinking: str,
                    execution_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create a new action for a workflow.
        """
        try:
            action_data = {
                'workflow_id': workflow_id,
                'action_type': action_type,  # Use action_type directly instead of action_type_id
                'description': description,
                'thinking': thinking,
                'execution_data': execution_data or {},
                'status': 'New'
            }
            
            # No longer trying to lookup action_type_id
                
            response = supabase.table('actions').insert(action_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating action: {e}")
            return None
    
    @staticmethod
    def update_action_status(action_id: str, status: str, 
                           execution_data: Optional[Dict] = None) -> bool:
        """
        Update an action's status and optionally its execution data.
        """
        try:
            update_data = {
                'status': status
            }
            
            if status in ['Complete', 'Error']:
                update_data['completed_at'] = 'now()'
                
            if execution_data:
                update_data['execution_data'] = execution_data
                
            response = supabase.table('actions').update(update_data).eq('id', action_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating action: {e}")
            return False
    
    @staticmethod
    def add_log(message: str, level: str = 'info', 
              user_id: Optional[str] = None,
              action_id: Optional[str] = None,
              details: Optional[Dict] = None) -> Optional[Dict]:
        """
        Add a new log entry.
        """
        try:
            log_data = {
                'message': message,
                'level': level,
                'details': details or {}
            }
            
            if user_id:
                log_data['user_id'] = user_id
                
            if action_id:
                log_data['action_id'] = action_id
                
            response = supabase.table('logs').insert(log_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error adding log: {e}")
            return None
        
    @staticmethod
    def complete_workflow(instance_id: str, context_data: Optional[Dict] = None) -> bool:
        """
        Mark a workflow as completed.
        
        Args:
            instance_id: ID of the workflow instance
            context_data: Final context data (optional)
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            update_data = {
                'status': 'completed',
                'completed_at': 'now()'
            }
            
            if context_data:
                update_data['context_data'] = context_data
                
            response = supabase.table('workflow_instances').update(update_data).eq('id', instance_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error completing workflow: {e}")
            raise    
    
    @staticmethod
    def get_latest_action_for_workflow(workflow_id: str) -> Optional[Dict]:
        """
        Get the most recent action for a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Latest action or None if no actions found
        """
        try:
            response = supabase.table('actions').select('*').eq('workflow_id', workflow_id) \
                              .order('created_at', desc=True).limit(1).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting latest action for workflow: {e}")
            raise    
        
    @staticmethod
    def get_actions_for_workflow(workflow_id: str, limit: int = 10) -> List[Dict]:
        """
        Get actions for a specific workflow.
        
        Args:
            workflow_id: ID of the workflow
            limit: Maximum number of actions to return
            
        Returns:
            List of actions
        """
        try:
            response = supabase.table('actions').select('*').eq('workflow_id', workflow_id) \
                              .order('created_at', desc=True).limit(limit).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting actions for workflow: {e}")
            raise    
        
    @staticmethod
    def get_workflow_instance(instance_id: str) -> Optional[Dict]:
        """
        Get a specific workflow instance.
        
        Args:
            instance_id: ID of the workflow instance
            
        Returns:
            Workflow instance or None if not found
        """
        try:
            response = supabase.table('workflow_instances').select('*').eq('id', instance_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting workflow instance: {e}")
            raise
        
    @staticmethod
    def get_workflow_instances_for_user(user_id: str, status: Optional[str] = None, 
                                      limit: int = 5) -> List[Dict]:
        """
        Get workflow instances for a user, optionally filtered by status.
        
        Args:
            user_id: ID of the user
            status: Status to filter by (optional)
            limit: Maximum number of instances to return
            
        Returns:
            List of workflow instances
        """
        try:
            query = supabase.table('workflow_instances').select('*').eq('user_id', user_id)
            
            if status:
                query = query.eq('status', status)
                
            response = query.order('started_at', desc=True).limit(limit).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting workflow instances for user: {e}")
            raise
        
    @staticmethod
    def get_full_workflow(instance_id: str) -> Optional[Dict]:
        """
        Get a workflow instance with its associated definition.
        
        Args:
            instance_id: ID of the workflow instance
            
        Returns:
            Combined workflow data or None if not found
        """
        try:
            # Get the workflow instance
            instance_response = supabase.table('workflow_instances').select('*').eq('id', instance_id).execute()
            if not instance_response.data or len(instance_response.data) == 0:
                logger.error(f"Workflow instance not found: {instance_id}")
                return None
                
            instance = instance_response.data[0]
            
            # Get the associated workflow definition
            definition_id = instance.get('definition_id')
            if not definition_id:
                logger.error(f"Workflow instance {instance_id} has no definition_id")
                return instance
                
            definition_response = supabase.table('workflow_definitions').select('*').eq('id', definition_id).execute()
            if not definition_response.data or len(definition_response.data) == 0:
                logger.error(f"Workflow definition not found: {definition_id}")
                return instance
                
            definition = definition_response.data[0]
            
            # Combine the data
            return {
                'id': instance_id,
                'instance_id': instance_id,
                'definition_id': definition_id,
                'name': definition.get('name'),
                'description': definition.get('description'),
                'steps': definition.get('steps', {}),
                'status': instance.get('status'),
                'current_step': instance.get('current_step'),
                'context_data': instance.get('context_data', {}),
                'user_id': instance.get('user_id'),
                'started_at': instance.get('started_at'),
                'completed_at': instance.get('completed_at')
            }
        except Exception as e:
            logger.error(f"Error getting full workflow: {e}")
            raise
    
    @staticmethod
    def error_workflow(instance_id: str, error_message: str, context_data: Optional[Dict] = None) -> bool:
        """
        Mark a workflow as errored with an error message.
        
        Args:
            instance_id: ID of the workflow instance
            error_message: Error message to store
            context_data: Final context data (optional)
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            update_data = {
                'status': 'error',
                'error_message': error_message,
                'completed_at': 'now()'
            }
            
            if context_data:
                update_data['context_data'] = context_data
                
            response = supabase.table('workflow_instances').update(update_data).eq('id', instance_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error marking workflow as errored: {e}")
            raise
    
    @staticmethod
    def get_entity_by_id(entity_id: str) -> Optional[Dict]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity or None if not found
        """
        try:
            response = supabase.table('entities').select('*').eq('id', entity_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting entity: {e}")
            raise
            
    @staticmethod
    def cancel_workflow(instance_id: str, reason: str = "Cancelled by system") -> bool:
        """
        Cancel a workflow instance.
        
        Args:
            instance_id: ID of the workflow instance
            reason: Reason for cancellation
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            update_data = {
                'status': 'cancelled',
                'completed_at': 'now()',
                'context_data': {
                    'cancellation_reason': reason
                }
            }
                
            response = supabase.table('workflow_instances').update(update_data).eq('id', instance_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            raise
        
    @staticmethod
    def update_message(message_id: str, workflow_id: str = None, content: str = None,
                    direction: str = None, metadata: Dict = None) -> bool:
        """
        Update an existing message.
        
        Args:
            message_id: ID of the message to update
            workflow_id: Optional new workflow ID
            content: Optional new message content
            direction: Optional new direction
            metadata: Optional new metadata
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            update_data = {}
            
            if workflow_id:
                update_data['related_workflow_id'] = workflow_id
                
            if content:
                update_data['message_content'] = content
                
            if direction:
                update_data['direction'] = direction
                
            if metadata:
                update_data['metadata'] = metadata
                
            if not update_data:
                return True  # Nothing to update
                
            response = supabase.table('messages').update(update_data).eq('id', message_id).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            return False

    @staticmethod
    def get_active_workflows_for_user(user_id: str) -> List[Dict]:
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
    def get_materialized_view(view_name: str) -> Optional[Dict]:
        """
        Get a materialized view by name.
        
        Args:
            view_name: Name of the materialized view
            
        Returns:
            View content or None if not found
        """
        try:
            # For MVP, return a simple placeholder
            # In a real implementation, you would query a views table
            # or generate the view on demand
            return {}
        except Exception as e:
            logger.error(f"Error getting materialized view: {e}")
            return None
        
    @staticmethod
    def create_entity(entity_type: str, entity_name: str, created_by: str,
                    parent_id: Optional[str] = None, attributes: Dict = None,
                    tags: List[str] = None, verification_status: str = 'unverified') -> Optional[str]:
        """
        Create a new entity in the hierarchical structure.
        """
        try:
            attributes = attributes or {}
            tags = tags or []
            
            # Call the database function
            response = supabase.rpc('create_entity', {
                'p_entity_type': entity_type,
                'p_entity_name': entity_name,
                'p_created_by': created_by,
                'p_parent_id': parent_id,
                'p_attributes': attributes,
                'p_tags': tags,
                'p_verification_status': verification_status
            }).execute()
            
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error creating entity: {e}")
            return None

    @staticmethod
    def update_entity_attributes(entity_id: str, attributes: Dict) -> bool:
        """
        Update an entity's attributes.
        """
        try:
            response = supabase.rpc('update_entity_attributes', {
                'p_entity_id': entity_id,
                'p_attributes': attributes
            }).execute()
            
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating entity attributes: {e}")
            return False
        
    @staticmethod
    def search_entities_with_context(search_text: str, entity_types: List[str] = None,
                                width: int = 2, limit: int = 10) -> Dict:
        """
        Search for entities with hierarchical context.
        """
        try:
            response = supabase.rpc('search_entities', {
                'p_search_text': search_text,
                'p_entity_types': entity_types,
                'p_width': width,
                'p_limit': limit
            }).execute()
            
            return response.data or {'results': [], 'count': 0}
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return {'error': str(e), 'results': [], 'count': 0}

    @staticmethod
    def get_entity_with_context(entity_id: str, max_depth: int = 2) -> Dict:
        """
        Get an entity with its hierarchical context.
        """
        try:
            response = supabase.rpc('get_entity_with_context', {
                'p_entity_id': entity_id,
                'p_max_depth': max_depth
            }).execute()
            
            return response.data or {}
        except Exception as e:
            logger.error(f"Error getting entity with context: {e}")
            return {'error': str(e)}