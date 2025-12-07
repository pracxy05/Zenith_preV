"""
WhatsApp Control Module
Uses pywhatkit for WhatsApp automation
"""

import logging
import pywhatkit
import time
from typing import Optional

logger = logging.getLogger(__name__)

class WhatsAppController:
    """WhatsApp message automation using pywhatkit"""
    
    def __init__(self):
        """Initialize WhatsApp controller"""
        logger.info("WhatsApp controller initialized")
    
    def send_message(self, contact: str, message: str) -> str:
        """
        Send WhatsApp message to a contact
        
        Args:
            contact: Phone number (with country code) or contact name
            message: Message to send
        
        Returns:
            Success message
        """
        try:
            # Get current time + 2 minutes for scheduling
            current_time = time.localtime()
            hour = current_time.tm_hour
            minute = current_time.tm_min + 2
            
            # Handle minute overflow
            if minute >= 60:
                hour += 1
                minute -= 60
            
            # Send message (opens WhatsApp Web)
            logger.info(f"Scheduling WhatsApp to {contact} at {hour}:{minute}")
            
            # If contact looks like a phone number, use sendwhatmsg
            if contact.startswith('+') or contact.replace('-', '').replace(' ', '').isdigit():
                pywhatkit.sendwhatmsg(contact, message, hour, minute, wait_time=15, tab_close=False)
            else:
                # For contact names, use instant send
                pywhatkit.sendwhatmsg_instantly(contact, message, wait_time=15, tab_close=False)
            
            logger.info(f"WhatsApp message sent to {contact}")
            return f"Message sent to {contact}"
            
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return f"Could not send WhatsApp: {str(e)}"
    
    def send_message_to_group(self, group_id: str, message: str) -> str:
        """
        Send message to WhatsApp group
        
        Args:
            group_id: Group ID (from WhatsApp Web URL)
            message: Message to send
        
        Returns:
            Success message
        """
        try:
            current_time = time.localtime()
            hour = current_time.tm_hour
            minute = current_time.tm_min + 2
            
            if minute >= 60:
                hour += 1
                minute -= 60
            
            pywhatkit.sendwhatmsg_to_group(group_id, message, hour, minute, wait_time=15, tab_close=False)
            
            logger.info(f"WhatsApp group message sent")
            return f"Message sent to group"
            
        except Exception as e:
            logger.error(f"WhatsApp group send error: {e}")
            return f"Could not send to group: {str(e)}"


# Global instance
_whatsapp_controller = None

def get_whatsapp_controller() -> WhatsAppController:
    """Get or create WhatsApp controller singleton"""
    global _whatsapp_controller
    if _whatsapp_controller is None:
        _whatsapp_controller = WhatsAppController()
    return _whatsapp_controller


# ============================================
# Public Functions for Function Registry
# ============================================

def send_whatsapp(contact: str, message: str) -> str:
    """Send WhatsApp message to contact"""
    controller = get_whatsapp_controller()
    return controller.send_message(contact, message)

def send_whatsapp_group(group_id: str, message: str) -> str:
    """Send WhatsApp message to group"""
    controller = get_whatsapp_controller()
    return controller.send_message_to_group(group_id, message)
