"""
Supabase client module for database operations.
"""

import json
from typing import Dict, List, Optional, Any, Union
import uuid

from supabase import create_client, Client

from app.config.settings import SUPABASE_URL, SUPABASE_KEY

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
            print(f"Error getting user: {e}")
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
            print(f"Error creating user: {e}")
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
            print(f"Error adding message: {e}")
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
            print(f"Error getting messages: {e}")
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
            print(f"Error creating/updating entity: {e}")
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
            print(f"Error searching entities: {e}")
            return []
    
    @staticmethod
    def get_or_create_workflow(name: str, user_id: str, 
                             description: Optional[str] = None,
                             steps: Optional[Dict] = None) -> Optional[Dict]:
        """
        Get an existing workflow or create a new one.
        """
        try:
            # For MVP 0.01, we simplify by creating a new workflow instance each time
            workflow_data = {
                'name': name,
                'description': description or f"Workflow {name}",
                'steps': steps or {},
                'user_id': user_id,
                'status': 'new',
                'started_at': 'now()'
            }
            
            response = supabase.table('workflows').insert(workflow_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error creating workflow: {e}")
            return None
    
    @staticmethod
    def update_workflow_status(workflow_id: str, status: str, 
                             current_step: Optional[str] = None,
                             context_data: Optional[Dict] = None) -> bool:
        """
        Update a workflow's status and optionally its current step and context.
        """
        try:
            update_data = {'status': status}
            if current_step:
                update_data['current_step'] = current_step
            if context_data:
                update_data['context_data'] = context_data
                
            response = supabase.table('workflows').update(update_data).eq('id', workflow_id).execute()
            return bool(response.data)
        except Exception as e:
            print(f"Error updating workflow: {e}")
            return False
    
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
                'action_type': action_type,
                'description': description,
                'thinking': thinking,
                'execution_data': execution_data or {},
                'status': 'New'
            }
            
            response = supabase.table('actions').insert(action_data).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error creating action: {e}")
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
            print(f"Error updating action: {e}")
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
            print(f"Error adding log: {e}")
            return None