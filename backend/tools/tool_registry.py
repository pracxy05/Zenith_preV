"""
Centralized Tool Registry - Production Grade v5.0
Registers ALL tools: System, Web, Spotify, File Control, WhatsApp

✅ UPDATED: Added open_whatsapp, open_spotify, play_downloaded_songs
Total: 26 functions across 5 categories
"""

import logging
from core.function_registry import FunctionRegistry

logger = logging.getLogger(__name__)


def register_all_tools(function_registry: FunctionRegistry) -> None:
    """
    Register all available tools with the function registry
    
    Categories:
        1. System Tools (3) - Time, battery, system info
        2. Web Control (7) - Browser automation, YouTube, WhatsApp open  ✅ +1
        3. Spotify Control (9) - Music playback, open, downloaded songs  ✅ +2
        4. File Control (5) - File explorer, file operations
        5. WhatsApp Control (2) - Messaging
    
    Args:
        function_registry: FunctionRegistry instance from assistant
    """
    tool_count = 0
    
    # ============================================
    # CATEGORY 1: SYSTEM TOOLS (3)
    # ============================================
    try:
        from tools.system import SystemTools
        
        system_tools = SystemTools()
        
        function_registry.register_function(
            "get_current_time",
            system_tools.get_current_time,
            "Get the current date and time"
        )
        
        function_registry.register_function(
            "get_system_info",
            system_tools.get_system_info,
            "Get system information (OS, CPU, memory)"
        )
        
        function_registry.register_function(
            "get_battery_status",
            system_tools.get_battery_status,
            "Get battery status and percentage"
        )
        
        tool_count += 3
        logger.info("✅ Registered 3 system tools")
        
    except Exception as e:
        logger.error(f"❌ Failed to register system tools: {e}")
    
    # ============================================
    # CATEGORY 2: WEB CONTROL (7) - ✅ Added open_whatsapp
    # ============================================
    try:
        from tools.web_control import (
            open_website,
            open_whatsapp,  # ✅ NEW - Opens WhatsApp Web/App
            search_google,
            search_youtube,
            play_youtube,
            youtube_pause,
            close_browser
        )
        
        function_registry.register_function(
            "open_website",
            open_website,
            "Open a website in Chrome browser (uses your profile)"
        )
        
        # ✅ NEW: Open WhatsApp
        function_registry.register_function(
            "open_whatsapp",
            open_whatsapp,
            "Open WhatsApp Web (via Chrome App or browser)"
        )
        
        function_registry.register_function(
            "search_google",
            search_google,
            "Search Google for information"
        )
        
        function_registry.register_function(
            "search_youtube",
            search_youtube,
            "Search for videos on YouTube"
        )
        
        function_registry.register_function(
            "play_youtube",
            play_youtube,
            "Play a YouTube video by search query"
        )
        
        function_registry.register_function(
            "youtube_pause",
            youtube_pause,
            "Pause or resume YouTube video"
        )
        
        function_registry.register_function(
            "close_browser",
            close_browser,
            "Close the Chrome browser"
        )
        
        tool_count += 7
        logger.info("✅ Registered 7 web control tools")
        
    except Exception as e:
        logger.error(f"❌ Failed to register web control tools: {e}")
    
    # ============================================
    # CATEGORY 3: SPOTIFY CONTROL (9) - ✅ Added open_spotify, play_downloaded_songs
    # ============================================
    try:
        from tools.spotify_control import (
            play_spotify,
            open_spotify,           # ✅ NEW - Just opens Spotify without playing
            play_downloaded_songs,  # ✅ NEW - Plays downloaded/offline songs
            pause_spotify,
            resume_spotify,
            next_track,
            previous_track,
            set_volume,
            current_track
        )
        
        function_registry.register_function(
            "play_spotify",
            play_spotify,
            "Play music on Spotify by song name, artist, or playlist"
        )
        
        # ✅ NEW: Just open Spotify
        function_registry.register_function(
            "open_spotify",
            open_spotify,
            "Open Spotify application without playing anything"
        )
        
        # ✅ NEW: Play downloaded songs (offline mode)
        function_registry.register_function(
            "play_downloaded_songs",
            play_downloaded_songs,
            "Play downloaded songs from Spotify (offline mode)"
        )
        
        function_registry.register_function(
            "pause_spotify",
            pause_spotify,
            "Pause Spotify playback"
        )
        
        function_registry.register_function(
            "resume_spotify",
            resume_spotify,
            "Resume Spotify playback"
        )
        
        function_registry.register_function(
            "next_track",
            next_track,
            "Skip to next track on Spotify"
        )
        
        function_registry.register_function(
            "previous_track",
            previous_track,
            "Go to previous track on Spotify"
        )
        
        function_registry.register_function(
            "set_volume",
            set_volume,
            "Set Spotify volume (0-100)"
        )
        
        function_registry.register_function(
            "current_track",
            current_track,
            "Get currently playing track information"
        )
        
        tool_count += 9
        logger.info("✅ Registered 9 Spotify tools")
        
    except ImportError as e:
        logger.warning(f"⚠️  Spotify tools import error: {e}")
        logger.warning("    Some Spotify functions may not be available")
    except Exception as e:
        logger.error(f"❌ Failed to register Spotify tools: {e}")
    
    # ============================================
    # CATEGORY 4: FILE CONTROL (5)
    # ============================================
    try:
        from tools.file_control import (
            open_file_explorer,
            search_files,
            open_file,
            create_folder,
            delete_file
        )
        
        function_registry.register_function(
            "open_file_explorer",
            open_file_explorer,
            "Open Windows File Explorer at specified location"
        )
        
        function_registry.register_function(
            "search_files",
            search_files,
            "Search for files by name in directory"
        )
        
        function_registry.register_function(
            "open_file",
            open_file,
            "Open a file with default application"
        )
        
        function_registry.register_function(
            "create_folder",
            create_folder,
            "Create a new folder"
        )
        
        function_registry.register_function(
            "delete_file",
            delete_file,
            "Delete a file (move to recycle bin)"
        )
        
        tool_count += 5
        logger.info("✅ Registered 5 file control tools")
        
    except ImportError:
        logger.warning("⚠️  File control tools not available (pywin32 not installed)")
        logger.warning("    Install: pip install pywin32")
    except Exception as e:
        logger.error(f"❌ Failed to register file tools: {e}")
    
    # ============================================
    # CATEGORY 5: WHATSAPP CONTROL (2)
    # Note: open_whatsapp is in web_control, these are for SENDING messages
    # ============================================
    try:
        from tools.whatsapp_control import send_whatsapp
        
        function_registry.register_function(
            "send_whatsapp",
            send_whatsapp,
            "Send a WhatsApp message to contact"
        )
        
        tool_count += 1
        logger.info("✅ Registered 1 WhatsApp messaging tool")
        
    except ImportError:
        logger.warning("⚠️  WhatsApp messaging not available (pywhatkit not installed)")
        logger.warning("    Install: pip install pywhatkit")
        logger.warning("    Note: 'Open WhatsApp' still works via Chrome App!")
    except Exception as e:
        logger.error(f"❌ Failed to register WhatsApp tools: {e}")
    
    # ============================================
    # SUMMARY
    # ============================================
    total_tools = len(function_registry.functions)
    
    logger.info(f"🎯 Total tools registered: {total_tools}/{tool_count}")
    
    if total_tools < tool_count:
        logger.warning(f"⚠️  Some tools failed to register ({tool_count - total_tools} missing)")
    
    # Print summary with updated counts
    print(f"🔧 Loaded {total_tools} assistant tools")
    print(f"   📊 System: 3 | 🌐 Web: 7 | 🎵 Spotify: 9 | 📁 Files: 5 | 💬 WhatsApp: 1+")
