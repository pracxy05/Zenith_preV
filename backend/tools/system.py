"""
System Control Module - v4.5
✅ FIXED: Wrapped in SystemTools class for tool_registry
✅ Added GPU control functions
"""

import logging
import platform
import psutil
import subprocess
import os
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global reference to speech recognizer (set by assistant)
_speech_recognizer = None

def set_speech_recognizer(recognizer):
    """Set global speech recognizer reference"""
    global _speech_recognizer
    _speech_recognizer = recognizer


# ============================================
# ✅ FIX: SystemTools Class Wrapper
# ============================================
class SystemTools:
    """
    System control tools - Time, battery, apps, GPU control
    All methods are static for easy tool registration
    """
    
    @staticmethod
    def get_current_time() -> str:
        """Get current date and time"""
        now = datetime.now()
        return now.strftime("Today is %A, %B %d, %Y. The time is %I:%M %p")
    
    @staticmethod
    def get_system_info() -> str:
        """Get system information"""
        try:
            info = {
                "OS": f"{platform.system()} {platform.release()}",
                "CPU": platform.processor(),
                "CPU Cores": psutil.cpu_count(),
                "RAM": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                "Disk": f"{psutil.disk_usage('/').total / (1024**3):.0f} GB"
            }
            
            result = "System Information:\n"
            for key, value in info.items():
                result += f"{key}: {value}\n"
            
            return result.strip()
        
        except Exception as e:
            return f"Could not get system info: {e}"
    
    @staticmethod
    def get_battery_status() -> str:
        """Get battery status"""
        try:
            battery = psutil.sensors_battery()
            
            if battery is None:
                return "No battery detected (desktop PC)"
            
            percent = battery.percent
            plugged = "plugged in" if battery.power_plugged else "on battery"
            
            if battery.power_plugged:
                return f"Battery: {percent}% ({plugged})"
            else:
                time_left = battery.secsleft // 60
                return f"Battery: {percent}% ({plugged}, {time_left} minutes left)"
        
        except Exception as e:
            return f"Could not get battery status: {e}"
    
    @staticmethod
    def open_calculator() -> str:
        """Open Windows Calculator"""
        try:
            subprocess.Popen("calc.exe")
            return "Calculator opened"
        except Exception as e:
            return f"Could not open calculator: {e}"
    
    @staticmethod
    def open_notepad() -> str:
        """Open Notepad"""
        try:
            subprocess.Popen("notepad.exe")
            return "Notepad opened"
        except Exception as e:
            return f"Could not open notepad: {e}"
    
    @staticmethod
    def open_paint() -> str:
        """Open Paint"""
        try:
            subprocess.Popen("mspaint.exe")
            return "Paint opened"
        except Exception as e:
            return f"Could not open paint: {e}"
    
    @staticmethod
    def open_task_manager() -> str:
        """Open Task Manager"""
        try:
            subprocess.Popen("taskmgr.exe")
            return "Task Manager opened"
        except Exception as e:
            return f"Could not open task manager: {e}"
    
    @staticmethod
    def open_control_panel() -> str:
        """Open Control Panel"""
        try:
            subprocess.Popen("control.exe")
            return "Control Panel opened"
        except Exception as e:
            return f"Could not open control panel: {e}"
    
    # ===== GPU CONTROL FUNCTIONS =====
    
    @staticmethod
    def switch_whisper_gpu(mode: str) -> str:
        """
        Switch Whisper between GPU and CPU
        
        Args:
            mode: 'gpu' or 'cpu'
        
        Returns:
            Status message
        """
        global _speech_recognizer
        
        if _speech_recognizer is None:
            return "Speech recognizer not initialized"
        
        try:
            if mode.lower() == 'gpu':
                success = _speech_recognizer.switch_to_gpu()
                if success:
                    return "Switched Whisper to GPU mode - faster processing!"
                else:
                    return "Could not switch to GPU - GPU not available or already on GPU"
            
            elif mode.lower() == 'cpu':
                success = _speech_recognizer.switch_to_cpu()
                if success:
                    return "Switched Whisper to CPU mode"
                else:
                    return "Could not switch to CPU - already on CPU"
            
            else:
                return "Invalid mode. Use 'gpu' or 'cpu'"
        
        except Exception as e:
            return f"GPU switch error: {e}"
    
    @staticmethod
    def get_whisper_device_info() -> str:
        """
        Get current Whisper device information
        
        Returns:
            Device info string
        """
        global _speech_recognizer
        
        if _speech_recognizer is None:
            return "Speech recognizer not initialized"
        
        try:
            info = _speech_recognizer.get_device_info()
            
            response = f"Whisper Device Info:\n"
            response += f"Model: {info['model']}\n"
            response += f"Current Device: {info['current_device'].upper()}\n"
            response += f"GPU Available: {'Yes' if info['gpu_available'] else 'No'}\n"
            
            if 'gpu_name' in info:
                response += f"GPU: {info['gpu_name']}\n"
                response += f"VRAM: {info['vram_allocated']} / {info['vram_total']}\n"
            
            return response.strip()
        
        except Exception as e:
            return f"Error getting device info: {e}"


# ============================================
# Standalone Functions (for backward compatibility)
# ============================================

def get_current_time() -> str:
    """Get current date and time"""
    return SystemTools.get_current_time()

def get_system_info() -> str:
    """Get system information"""
    return SystemTools.get_system_info()

def get_battery_status() -> str:
    """Get battery status"""
    return SystemTools.get_battery_status()

def open_calculator() -> str:
    """Open Windows Calculator"""
    return SystemTools.open_calculator()

def open_notepad() -> str:
    """Open Notepad"""
    return SystemTools.open_notepad()

def open_paint() -> str:
    """Open Paint"""
    return SystemTools.open_paint()

def open_task_manager() -> str:
    """Open Task Manager"""
    return SystemTools.open_task_manager()

def open_control_panel() -> str:
    """Open Control Panel"""
    return SystemTools.open_control_panel()

def switch_whisper_gpu(mode: str) -> str:
    """Switch Whisper between GPU and CPU"""
    return SystemTools.switch_whisper_gpu(mode)

def get_whisper_device_info() -> str:
    """Get current Whisper device information"""
    return SystemTools.get_whisper_device_info()
