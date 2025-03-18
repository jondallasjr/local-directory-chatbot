"""
Twilio handler for WhatsApp integration.
"""

from typing import Dict, Any
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from app.config.settings import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from app.workflows.action_handler import ActionHandler
from app.database.supabase_client import SupabaseDB

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class TwilioHandler:
    """
    Handles WhatsApp integration via Twilio.
    """
    
    @staticmethod
    def process_webhook(request_data: Dict[str, Any]) -> str:
        """
        Process incoming webhook from Twilio.
        
        Args:
            request_data: Data from the webhook request
            
        Returns:
            TwiML response
        """
        # Extract message details
        incoming_msg = request_data.get('Body', '')
        sender_phone = request_data.get('From', '')
        
        # Clean phone number (remove WhatsApp: prefix if present)
        if sender_phone.startswith('whatsapp:'):
            sender_phone = sender_phone[9:]
        
        # Log the incoming message
        SupabaseDB.add_log(
            message=f"Received message from {sender_phone}",
            level='info',
            details={
                "message": incoming_msg
            }
        )
        
        # Process the message
        response = ActionHandler.process_incoming_message(
            phone_number=sender_phone,
            message_content=incoming_msg
        )
        
        # Create TwiML response
        twiml = MessagingResponse()
        
        if response.get('message'):
            twiml.message(response['message'])
        
        return str(twiml)
    
    @staticmethod
    def send_message(to_phone: str, message: str) -> bool:
        """
        Send a WhatsApp message via Twilio.
        
        Args:
            to_phone: Recipient's phone number
            message: Message content
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            # Ensure WhatsApp format for the phone number
            if not to_phone.startswith('whatsapp:'):
                to_phone = f"whatsapp:{to_phone}"
                
            # Ensure WhatsApp format for the from number
            from_phone = TWILIO_PHONE_NUMBER
            if not from_phone.startswith('whatsapp:'):
                from_phone = f"whatsapp:{from_phone}"
            
            # Send the message
            message = twilio_client.messages.create(
                body=message,
                from_=from_phone,
                to=to_phone
            )
            
            # Log success
            SupabaseDB.add_log(
                message=f"Sent message to {to_phone}",
                level='info',
                details={
                    "message_sid": message.sid,
                    "message": message.body
                }
            )
            
            return True
            
        except Exception as e:
            # Log error
            SupabaseDB.add_log(
                message=f"Failed to send message to {to_phone}",
                level='error',
                details={
                    "error": str(e),
                    "message": message
                }
            )
            
            return False