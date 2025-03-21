"""
API endpoints for the Directory Chatbot.
"""

from fastapi import FastAPI, Request, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
import traceback
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import logging

# Updated import - remove ACTION_TYPES and DEFAULT_WORKFLOWS
from app.workflows.action_handler import ActionHandler
from app.api.twilio_handler import TwilioHandler
from app.database.supabase_client import SupabaseDB, supabase

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Directory Chatbot API",
    description="API for the Community Directory Chatbot",
    version="0.01"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API models
class MessageRequest(BaseModel):
    phone_number: str
    message: str

class EntityRequest(BaseModel):
    entity_type: str
    entity_name: str
    created_by: str
    attributes: Dict = {}
    tags: List[str] = []
    verification_status: str = "unverified"
    metadata: Dict = {}

# Simple API key validation for MVP
def verify_api_key(x_api_key: str = Header(None)):
    """
    Simple API key validation for developer dashboard.
    For MVP 0.01, we use a hardcoded key.
    """
    if x_api_key != "dev-dashboard-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "version": "0.01"}

@app.post("/api/webhook/twilio")
async def twilio_webhook(request: Request):
    """
    Webhook endpoint for Twilio WhatsApp messages.
    """
    try:
        # Parse form data
        form_data = await request.form()
        data = dict(form_data)
        
        # Process the webhook
        twiml_response = TwilioHandler.process_webhook(data)
        
        # Return TwiML response
        return PlainTextResponse(content=twiml_response, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error processing Twilio webhook: {str(e)}")
        # Return a valid TwiML response even in case of error
        return PlainTextResponse(
            content="<?xml version='1.0' encoding='UTF-8'?><Response></Response>",
            media_type="application/xml"
        )

@app.post("/api/simulate/message")
async def simulate_message(
    data: MessageRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Endpoint for simulating a message in the developer dashboard.
    """
    try:
        # Process the message
        response = ActionHandler.process_incoming_message(
            phone_number=data.phone_number,
            message_content=data.message
        )
        
        # Get the latest action for this user/message
        user = SupabaseDB.get_user_by_phone(data.phone_number)
        if user:
            # Get latest workflow for this user
            workflow_response = supabase.table('workflow_instances') \
                .select('*') \
                .eq('user_id', user['id']) \
                .order('started_at', desc=True) \
                .limit(1) \
                .execute()
            
            workflow = workflow_response.data[0] if workflow_response.data else None
            
            if workflow:
                # Get latest action for this workflow
                action_response = supabase.table('actions') \
                    .select('*') \
                    .eq('workflow_id', workflow['id']) \
                    .order('created_at', desc=True) \
                    .limit(1) \
                    .execute()
                
                action = action_response.data[0] if action_response.data else None
                
                if action:
                    # Enhance the response with action details
                    response.update({
                        'thinking': action.get('thinking', ''),
                        'step': action.get('description', ''),
                        'action': action.get('action_type', ''),
                        'context': workflow.get('context_data', {})
                    })
        
        return response
    except Exception as e:
        logger.error(f"Error simulating message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users(
    api_key: str = Depends(verify_api_key),
    limit: int = Query(50, ge=1, le=100)
):
    """
    Get all users for the developer dashboard.
    """
    try:
        response = supabase.table('users') \
                          .select('*') \
                          .order('created_at', desc=True) \
                          .limit(limit) \
                          .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/messages/{user_id}")
async def get_user_messages(
    user_id: str,
    limit: int = Query(20, ge=1, le=100),
    api_key: str = Depends(verify_api_key)
):
    """
    Get messages for a specific user.
    """
    try:
        response = supabase.table('messages') \
                          .select('*') \
                          .eq('user_id', user_id) \
                          .order('timestamp', desc=True) \
                          .limit(limit) \
                          .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting messages for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows/{user_id}")
async def get_user_workflows(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """
    Get workflows for a specific user.
    """
    try:
        response = supabase.table('workflow_instances') \
                          .select('*') \
                          .eq('user_id', user_id) \
                          .order('created_at', desc=True) \
                          .limit(limit) \
                          .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting workflows for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/actions/{workflow_id}")
async def get_workflow_actions(
    workflow_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get actions for a specific workflow.
    """
    try:
        response = supabase.table('actions') \
                          .select('*') \
                          .eq('workflow_id', workflow_id) \
                          .order('created_at', desc=False) \
                          .execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting actions for workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(
    level: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    api_key: str = Depends(verify_api_key)
):
    """
    Get recent system logs.
    """
    try:
        query = supabase.table('logs').select('*')
        
        if level:
            query = query.eq('level', level)
            
        query = query.order('timestamp', desc=True).limit(limit)
        response = query.execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/entities")
async def get_entities(
    entity_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    api_key: str = Depends(verify_api_key)
):
    """
    Search for entities.
    """
    try:
        query = supabase.table('entities').select('*')
        
        if entity_type:
            query = query.eq('entity_type', entity_type)
            
        if search:
            query = query.ilike('entity_name', f'%{search}%')
            
        response = query.limit(limit).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/entities")
async def create_entity(
    entity: EntityRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a new entity.
    """
    try:
        result = SupabaseDB.create_or_update_entity(
            entity_type=entity.entity_type,
            entity_name=entity.entity_name,
            created_by=entity.created_by,
            attributes=entity.attributes,
            verification_status=entity.verification_status,
            metadata=entity.metadata,
            tags=entity.tags
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create entity")
            
        return result
    except Exception as e:
        logger.error(f"Error creating entity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/action-types")
async def get_action_types(
    api_key: str = Depends(verify_api_key)
):
    """
    Get available action types.
    """
    try:
        # Use the ActionHandler to get action types from database
        action_types = ActionHandler.get_action_types()
        return {"action_types": action_types}
    except Exception as e:
        logger.error(f"Error getting action types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows")
async def get_workflows(
    api_key: str = Depends(verify_api_key)
):
    """
    Get available workflow definitions.
    """
    try:
        # Use the ActionHandler to get workflow definitions from database
        workflows = ActionHandler.get_workflow_definitions()
        return {"workflows": workflows}
    except Exception as e:
        logger.error(f"Error getting workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/healthcheck")
async def healthcheck():
    """
    Health check endpoint.
    """
    # Check database connection
    try:
        # Try a simple database query
        response = supabase.table('logs').select('id').limit(1).execute()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "version": "0.01",
        "database": db_status,
        "timestamp": supabase.table('_rpc').select().execute().data
    }
    
# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler with detailed error information.
    """
    # Get detailed traceback
    tb_frames = traceback.extract_tb(exc.__traceback__)
    
    # Find the most relevant frame (closest to the error)
    relevant_frame = tb_frames[-1] if tb_frames else None
    
    # Extract location information
    if relevant_frame:
        file = relevant_frame.filename.split('/')[-1]  # Just the filename
        line = relevant_frame.lineno
        function = relevant_frame.name
        location = f"{file}:{line} in {function}()"
    else:
        location = "unknown location"
    
    # Log the error
    logger.error(f"Unhandled exception at {location}: {str(exc)}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "message": "Sorry, I encountered an error while processing your message. Please try again later.",
            "error": f"{str(exc)} at {location}",
            "debug_info": traceback.format_exc() if request.query_params.get("debug") == "true" else None
        }
    )