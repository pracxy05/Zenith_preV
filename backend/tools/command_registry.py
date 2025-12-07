"""
Command Registry - Voice Command to Function Mapping
Centralized command parsing and intent detection
Version 5.0 - ROBUST: Fixed YouTube/Spotify conflicts + WhatsApp support

🔧 KEY FIXES:
1. Parameterized patterns checked BEFORE direct commands for media
2. Exact match required for conflicting keywords (youtube, spotify)
3. Better query extraction (removes all noise words)
4. WhatsApp open commands added
"""

import logging
import re
from typing import Optional, Dict, Any, Tuple, List
from core.function_registry import FunctionRegistry

logger = logging.getLogger(__name__)


class CommandRegistry:
    """
    Handles all voice command parsing and intent mapping
    Maps natural language to function calls with parameters
    
    🔥 MATCHING ORDER (Important!):
    1. HIGH-PRIORITY parameterized patterns (YouTube play, Spotify play)
    2. Exact-match direct commands
    3. Contains-match direct commands (for non-conflicting keywords)
    4. LOW-PRIORITY parameterized patterns
    5. Fallback to LLM
    """
    
    def __init__(self, function_registry: FunctionRegistry):
        """
        Initialize command registry
        
        Args:
            function_registry: FunctionRegistry instance with all registered tools
        """
        self.function_registry = function_registry
        
        # ============================================
        # EXACT MATCH ONLY KEYWORDS
        # These keywords will ONLY match if the entire command equals them
        # Prevents "play sunflower on youtube" from matching "youtube" -> open youtube
        # ============================================
        self.exact_match_only = {
            "youtube", "spotify", "google", "gmail", "facebook", 
            "twitter", "instagram", "reddit", "github", "netflix", "amazon"
        }
        
        # ============================================
        # DIRECT COMMANDS (no parameters needed)
        # ============================================
        self.direct_commands = {
            # ===== TIME & DATE =====
            "time": "get_current_time",
            "what time": "get_current_time",
            "current time": "get_current_time",
            "what's the time": "get_current_time",
            "whats the time": "get_current_time",
            "tell me the time": "get_current_time",
            "what time is it": "get_current_time",
            "date": "get_current_time",
            "what date": "get_current_time",
            "today's date": "get_current_time",
            "todays date": "get_current_time",
            "what's the date": "get_current_time",
            "whats the date": "get_current_time",
            "what is the date": "get_current_time",
            "tell me the date": "get_current_time",
            
            # ===== SYSTEM INFO =====
            "system info": "get_system_info",
            "system information": "get_system_info",
            "computer info": "get_system_info",
            "pc info": "get_system_info",
            "system details": "get_system_info",
            "computer details": "get_system_info",
            
            # ===== BATTERY =====
            "battery": "get_battery_status",
            "battery status": "get_battery_status",
            "battery level": "get_battery_status",
            "how much battery": "get_battery_status",
            "battery percentage": "get_battery_status",
            "check battery": "get_battery_status",
            
            # ===== SPOTIFY LAUNCHER =====
            "open spotify": "open_spotify",
            "launch spotify": "open_spotify",
            "start spotify": "open_spotify",
            "run spotify": "open_spotify",
            
            # ===== WHATSAPP LAUNCHER ===== ✅ NEW
            "open whatsapp": "open_whatsapp",
            "launch whatsapp": "open_whatsapp",
            "start whatsapp": "open_whatsapp",
            "whatsapp open": "open_whatsapp",
            "open whats app": "open_whatsapp",
            
            # ===== GPU CONTROL =====
            "switch to gpu": ("switch_whisper_gpu", {"mode": "gpu"}),
            "use gpu": ("switch_whisper_gpu", {"mode": "gpu"}),
            "enable gpu": ("switch_whisper_gpu", {"mode": "gpu"}),
            "gpu mode": ("switch_whisper_gpu", {"mode": "gpu"}),
            "switch to cpu": ("switch_whisper_gpu", {"mode": "cpu"}),
            "use cpu": ("switch_whisper_gpu", {"mode": "cpu"}),
            "disable gpu": ("switch_whisper_gpu", {"mode": "cpu"}),
            "cpu mode": ("switch_whisper_gpu", {"mode": "cpu"}),
            "device info": "get_whisper_device_info",
            "check gpu": "get_whisper_device_info",
            "gpu status": "get_whisper_device_info",
            "whisper info": "get_whisper_device_info",
            
            # ===== SYSTEM APP LAUNCHERS =====
            "open calculator": "open_calculator",
            "calculator": "open_calculator",
            "launch calculator": "open_calculator",
            "start calculator": "open_calculator",
            "calc": "open_calculator",
            
            "open notepad": "open_notepad",
            "notepad": "open_notepad",
            "launch notepad": "open_notepad",
            "start notepad": "open_notepad",
            "text editor": "open_notepad",
            
            "open paint": "open_paint",
            "paint": "open_paint",
            "launch paint": "open_paint",
            "start paint": "open_paint",
            "ms paint": "open_paint",
            
            "open task manager": "open_task_manager",
            "task manager": "open_task_manager",
            "launch task manager": "open_task_manager",
            "start task manager": "open_task_manager",
            
            "open control panel": "open_control_panel",
            "control panel": "open_control_panel",
            "launch control panel": "open_control_panel",
            "start control panel": "open_control_panel",
            
            # ===== WEB - DIRECT URLS =====
            # Note: Single word like "youtube" requires EXACT match (handled in process_command)
            "open google": ("open_website", {"url": "google.com"}),
            "google": ("open_website", {"url": "google.com"}),  # Exact match only
            "launch google": ("open_website", {"url": "google.com"}),
            
            "open youtube": ("open_website", {"url": "youtube.com"}),
            "youtube": ("open_website", {"url": "youtube.com"}),  # Exact match only
            "launch youtube": ("open_website", {"url": "youtube.com"}),
            
            "open gmail": ("open_website", {"url": "gmail.com"}),
            "gmail": ("open_website", {"url": "gmail.com"}),
            "launch gmail": ("open_website", {"url": "gmail.com"}),
            
            "open facebook": ("open_website", {"url": "facebook.com"}),
            "facebook": ("open_website", {"url": "facebook.com"}),
            
            "open twitter": ("open_website", {"url": "twitter.com"}),
            "twitter": ("open_website", {"url": "twitter.com"}),
            "open x": ("open_website", {"url": "x.com"}),
            
            "open instagram": ("open_website", {"url": "instagram.com"}),
            "instagram": ("open_website", {"url": "instagram.com"}),
            
            "open reddit": ("open_website", {"url": "reddit.com"}),
            "reddit": ("open_website", {"url": "reddit.com"}),
            
            "open github": ("open_website", {"url": "github.com"}),
            "github": ("open_website", {"url": "github.com"}),
            
            "open linkedin": ("open_website", {"url": "linkedin.com"}),
            "linkedin": ("open_website", {"url": "linkedin.com"}),
            
            "open netflix": ("open_website", {"url": "netflix.com"}),
            "netflix": ("open_website", {"url": "netflix.com"}),
            
            "open amazon": ("open_website", {"url": "amazon.in"}),
            "amazon": ("open_website", {"url": "amazon.in"}),
            
            "open flipkart": ("open_website", {"url": "flipkart.com"}),
            "flipkart": ("open_website", {"url": "flipkart.com"}),
            
            "open stackoverflow": ("open_website", {"url": "stackoverflow.com"}),
            "stack overflow": ("open_website", {"url": "stackoverflow.com"}),
            
            "open chatgpt": ("open_website", {"url": "chat.openai.com"}),
            "chatgpt": ("open_website", {"url": "chat.openai.com"}),
            
            "open perplexity": ("open_website", {"url": "perplexity.ai"}),
            "perplexity": ("open_website", {"url": "perplexity.ai"}),
            
            # ===== BROWSER CONTROL =====
            "close browser": "close_browser",
            "close chrome": "close_browser",
            "quit browser": "close_browser",
            "exit browser": "close_browser",
            "close the browser": "close_browser",
            "shut down browser": "close_browser",
            
            # ===== YOUTUBE CONTROL (no search) =====
            "pause youtube": "youtube_pause",
            "pause video": "youtube_pause",
            "resume youtube": "youtube_pause",
            "resume video": "youtube_pause",
            "play pause": "youtube_pause",
            "play pause youtube": "youtube_pause",
            "toggle youtube": "youtube_pause",
            
            # ===== SPOTIFY CONTROL =====
            "pause music": "pause_spotify",
            "pause spotify": "pause_spotify",
            "pause song": "pause_spotify",
            "stop music": "pause_spotify",
            "stop spotify": "pause_spotify",
            "stop the music": "pause_spotify",
            
            "resume music": "resume_spotify",
            "resume spotify": "resume_spotify",
            "continue music": "resume_spotify",
            "unpause spotify": "resume_spotify",
            "unpause music": "resume_spotify",
            
            "next song": "next_track",
            "skip song": "next_track",
            "next track": "next_track",
            "skip": "next_track",
            "skip track": "next_track",
            "next": "next_track",
            
            "previous song": "previous_track",
            "previous track": "previous_track",
            "go back": "previous_track",
            "last song": "previous_track",
            "back": "previous_track",
            "previous": "previous_track",
            
            "what's playing": "current_track",
            "whats playing": "current_track",
            "current song": "current_track",
            "now playing": "current_track",
            "what song is this": "current_track",
            "what song": "current_track",
            "song name": "current_track",
            
            # ✅ Offline mode support
            "play downloaded songs": "play_downloaded_songs",
            "play my downloaded songs": "play_downloaded_songs",
            "play downloaded music": "play_downloaded_songs",
            "play offline": "play_downloaded_songs",
            "downloaded songs": "play_downloaded_songs",
            "offline music": "play_downloaded_songs",
            "play my downloads": "play_downloaded_songs",
            "play downloaded": "play_downloaded_songs",
            
            # ===== FILE EXPLORER =====
            "open downloads": ("open_file_explorer", {"location": "downloads"}),
            "downloads": ("open_file_explorer", {"location": "downloads"}),
            "downloads folder": ("open_file_explorer", {"location": "downloads"}),
            
            "open documents": ("open_file_explorer", {"location": "documents"}),
            "documents": ("open_file_explorer", {"location": "documents"}),
            "documents folder": ("open_file_explorer", {"location": "documents"}),
            
            "open desktop": ("open_file_explorer", {"location": "desktop"}),
            "desktop folder": ("open_file_explorer", {"location": "desktop"}),
            
            "open pictures": ("open_file_explorer", {"location": "pictures"}),
            "pictures": ("open_file_explorer", {"location": "pictures"}),
            "pictures folder": ("open_file_explorer", {"location": "pictures"}),
            
            "open music folder": ("open_file_explorer", {"location": "music"}),
            "music folder": ("open_file_explorer", {"location": "music"}),
            
            "open videos": ("open_file_explorer", {"location": "videos"}),
            "videos folder": ("open_file_explorer", {"location": "videos"}),
            
            "open file explorer": ("open_file_explorer", {"location": None}),
            "file explorer": ("open_file_explorer", {"location": None}),
            "open files": ("open_file_explorer", {"location": None}),
            "explorer": ("open_file_explorer", {"location": None}),
        }
        
        # ============================================
        # PARAMETERIZED COMMANDS (require extraction)
        # Priority: Higher number = checked first
        # ============================================
        self.parameterized_patterns = [
            # ===== YOUTUBE PLAY - HIGHEST PRIORITY (100) =====
            # Must be checked BEFORE direct "youtube" command
            {
                "patterns": [
                    "play on youtube", "play youtube", "youtube play",
                    "on youtube", "watch on youtube", "watch youtube",
                    "play video", "play the video"
                ],
                "function": "play_youtube",
                "param_name": "query",
                "remove_keywords": [
                    "play on youtube", "play youtube", "youtube play",
                    "on youtube", "watch on youtube", "watch youtube",
                    "play video", "play the video", "play", "video", 
                    "youtube", "watch", "the", "a", "on"
                ],
                "priority": 100  # HIGHEST - Check before any youtube direct command
            },
            
            # ===== YOUTUBE SEARCH (95) =====
            {
                "patterns": [
                    "search youtube", "youtube search", "search on youtube",
                    "find on youtube", "look up on youtube", "youtube for"
                ],
                "function": "search_youtube",
                "param_name": "query",
                "remove_keywords": [
                    "search youtube", "youtube search", "search on youtube",
                    "find on youtube", "look up on youtube", "youtube for",
                    "search", "find", "look up", "on youtube", "youtube", "for"
                ],
                "priority": 95
            },
            
            # ===== SPOTIFY PLAY (90) =====
            # Must be checked BEFORE direct "spotify" command
            {
                "patterns": [
                    "play on spotify", "play spotify", "spotify play",
                    "on spotify", "play song", "play music", "play the song",
                    "play me"
                ],
                "function": "play_spotify",
                "param_name": "query",
                "remove_keywords": [
                    "play on spotify", "play spotify", "spotify play",
                    "on spotify", "play song", "play music", "play the song",
                    "play me", "play", "song", "music", "spotify", 
                    "the", "a", "on", "me", "by"
                ],
                "priority": 90
            },
            
            # ===== GENERIC PLAY (80) =====
            # "Play Sunflower" without specifying platform -> Spotify
            {
                "patterns": ["play"],
                "function": "play_spotify",
                "param_name": "query",
                "remove_keywords": ["play", "the", "a", "song", "music"],
                "priority": 80,
                "min_query_length": 2  # Must have at least 2 chars after removing keywords
            },
            
            # ===== GOOGLE SEARCH (70) =====
            {
                "patterns": [
                    "search google", "google search", "search for",
                    "google for", "look up", "find on google", "search"
                ],
                "function": "search_google",
                "param_name": "query",
                "remove_keywords": [
                    "search google", "google search", "search for",
                    "google for", "look up", "find on google", 
                    "google", "search", "for", "find"
                ],
                "priority": 70
            },
            
            # ===== OPEN WEBSITE (60) =====
            {
                "patterns": [
                    "open website", "go to website", "navigate to",
                    "browse to", "visit website", "go to", "visit"
                ],
                "function": "open_website",
                "param_name": "url",
                "remove_keywords": [
                    "open website", "go to website", "navigate to",
                    "browse to", "visit website", "go to", "visit",
                    "website", "the"
                ],
                "priority": 60
            },
            
            # ===== OPEN URL DIRECTLY (50) =====
            {
                "patterns": ["open"],
                "function": "open_website",
                "param_name": "url",
                "remove_keywords": ["open", "the"],
                "requires_domain": True,  # Must contain .com/.org/etc
                "priority": 50
            },
            
            # ===== SPOTIFY VOLUME (45) =====
            {
                "patterns": [
                    "set volume to", "volume to", "change volume to",
                    "set volume", "volume", "adjust volume to"
                ],
                "function": "set_volume",
                "param_name": "volume",
                "remove_keywords": [
                    "set volume to", "volume to", "change volume to",
                    "set volume", "volume", "adjust volume to",
                    "to", "percent", "%", "at"
                ],
                "parse_number": True,  # Extract number from text
                "priority": 45
            },
            
            # ===== FILE SEARCH (40) =====
            {
                "patterns": [
                    "search files for", "find file", "search for file",
                    "look for file", "search file", "find"
                ],
                "function": "search_files",
                "param_name": "filename",
                "remove_keywords": [
                    "search files for", "find file", "search for file",
                    "look for file", "search files", "search for",
                    "search file", "find", "file", "the"
                ],
                "priority": 40
            },
            
            # ===== OPEN FILE (35) =====
            {
                "patterns": ["open file", "open document"],
                "function": "open_file",
                "param_name": "filename",
                "remove_keywords": ["open file", "open document", "open", "file", "document", "the"],
                "priority": 35
            },
            
            # ===== CREATE FOLDER (30) =====
            {
                "patterns": [
                    "create folder", "make folder", "new folder",
                    "create directory", "make directory", "new directory"
                ],
                "function": "create_folder",
                "param_name": "folder_name",
                "remove_keywords": [
                    "create folder", "make folder", "new folder",
                    "create directory", "make directory", "new directory",
                    "create", "make", "new", "called", "named", "folder", "directory"
                ],
                "priority": 30
            },
            
            # ===== DELETE FILE (25) =====
            {
                "patterns": [
                    "delete file", "remove file", "trash file", "delete", "remove"
                ],
                "function": "delete_file",
                "param_name": "filename",
                "remove_keywords": [
                    "delete file", "remove file", "trash file",
                    "delete", "remove", "trash", "file", "the"
                ],
                "priority": 25
            },
            
            # ===== WHATSAPP SEND (20) =====
            {
                "patterns": [
                    "send whatsapp to", "whatsapp message to",
                    "message on whatsapp to", "send message to",
                    "whatsapp to", "message to"
                ],
                "function": "send_whatsapp",
                "param_name": "contact",
                "remove_keywords": [
                    "send whatsapp to", "whatsapp message to",
                    "message on whatsapp to", "send message to",
                    "whatsapp to", "message to", "whatsapp", "message", "send", "to"
                ],
                "extract_message": True,
                "message_keywords": ["saying", "tell them", "message", "that", "say"],
                "priority": 20
            },
        ]
        
        # Sort patterns by priority (highest first)
        self.parameterized_patterns.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    def process_command(self, user_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Process user command and return function name + parameters
        
        🔥 MATCHING ORDER:
        1. High-priority parameterized patterns (YouTube, Spotify with queries)
        2. Exact-match direct commands
        3. Contains-match direct commands
        4. Low-priority parameterized patterns
        
        Args:
            user_text: Raw user speech text
        
        Returns:
            Tuple of (function_name, kwargs) or None if no match
        """
        user_lower = user_text.lower().strip()
        
        # Remove punctuation for better matching
        user_lower = user_lower.rstrip('.,!?')
        
        # ============================================
        # STEP 1: Check HIGH-PRIORITY Parameterized Commands
        # (Priority >= 80: YouTube play, Spotify play, etc.)
        # This ensures "play sunflower on youtube" is handled correctly
        # ============================================
        result = self._check_parameterized_patterns(user_lower, min_priority=80)
        if result:
            return result
        
        # ============================================
        # STEP 2: Check EXACT MATCH Direct Commands
        # ============================================
        if user_lower in self.direct_commands:
            mapping = self.direct_commands[user_lower]
            if isinstance(mapping, tuple):
                func_name, params = mapping
                logger.info(f"✅ Direct command (exact): {func_name} with params {params}")
                return (func_name, params)
            else:
                logger.info(f"✅ Direct command (exact): {mapping}")
                return (mapping, {})
        
        # ============================================
        # STEP 3: Check CONTAINS Direct Commands
        # But SKIP exact_match_only keywords unless exact match
        # ============================================
        for keyword, mapping in self.direct_commands.items():
            # Skip single words that need exact match
            if keyword in self.exact_match_only:
                continue
            
            # Check if keyword is in user text
            if keyword in user_lower:
                if isinstance(mapping, tuple):
                    func_name, params = mapping
                    logger.info(f"✅ Direct command (contains): {func_name} with params {params}")
                    return (func_name, params)
                else:
                    logger.info(f"✅ Direct command (contains): {mapping}")
                    return (mapping, {})
        
        # ============================================
        # STEP 4: Check REMAINING Parameterized Commands
        # (Priority < 80)
        # ============================================
        result = self._check_parameterized_patterns(user_lower, max_priority=79)
        if result:
            return result
        
        # ============================================
        # STEP 5: No Command Match - Will fall back to LLM
        # ============================================
        logger.debug(f"❌ No command match for: {user_text}")
        return None
    
    def _check_parameterized_patterns(
        self, 
        user_lower: str, 
        min_priority: int = 0, 
        max_priority: int = 1000
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Check parameterized patterns within priority range
        
        Args:
            user_lower: Lowercase user text
            min_priority: Minimum priority to check (inclusive)
            max_priority: Maximum priority to check (inclusive)
        
        Returns:
            Tuple of (function_name, kwargs) or None
        """
        for pattern_config in self.parameterized_patterns:
            priority = pattern_config.get("priority", 0)
            
            # Skip if outside priority range
            if priority < min_priority or priority > max_priority:
                continue
            
            patterns = pattern_config["patterns"]
            function = pattern_config["function"]
            param_name = pattern_config["param_name"]
            remove_keywords = pattern_config["remove_keywords"]
            requires_domain = pattern_config.get("requires_domain", False)
            parse_number = pattern_config.get("parse_number", False)
            extract_message = pattern_config.get("extract_message", False)
            min_query_length = pattern_config.get("min_query_length", 1)
            
            # Check if any pattern matches
            for pattern in patterns:
                if pattern in user_lower:
                    # ===== SPECIAL: WhatsApp Message Extraction =====
                    if extract_message:
                        result = self._extract_whatsapp_message(user_lower, pattern_config)
                        if result:
                            return result
                        continue
                    
                    # Extract parameter by removing keywords
                    param_value = self._extract_query(user_lower, remove_keywords)
                    
                    # Validate parameter length
                    if len(param_value) < min_query_length:
                        logger.debug(f"Query too short after extraction: '{param_value}'")
                        continue
                    
                    # ===== NUMBER PARSING (for volume, etc.) =====
                    if parse_number:
                        number = self._extract_number(param_value)
                        if number is not None:
                            logger.info(f"✅ Parameterized command: {function}({param_name}={number})")
                            return (function, {param_name: number})
                        else:
                            continue
                    
                    # ===== DOMAIN VALIDATION (for URLs) =====
                    if requires_domain:
                        if not self._is_valid_domain(param_value):
                            logger.debug(f"Not a valid domain: {param_value}")
                            continue
                    
                    logger.info(f"✅ Parameterized command: {function}({param_name}={param_value})")
                    return (function, {param_name: param_value})
        
        return None
    
    def _extract_query(self, text: str, remove_keywords: List[str]) -> str:
        """
        Extract the query by removing all keywords
        
        Args:
            text: User text (lowercase)
            remove_keywords: Keywords to remove
        
        Returns:
            Cleaned query string
        """
        result = text
        
        # Sort keywords by length (longest first) to avoid partial replacements
        sorted_keywords = sorted(remove_keywords, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            result = result.replace(keyword, " ")
        
        # Clean up extra spaces
        result = " ".join(result.split())
        
        return result.strip()
    
    def _extract_whatsapp_message(
        self, 
        user_text: str, 
        config: dict
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extract contact and message from WhatsApp command
        
        Example: "send whatsapp to John saying Hello how are you"
                 -> contact: "John", message: "Hello how are you"
        
        Args:
            user_text: User command
            config: Pattern configuration
        
        Returns:
            Tuple of (function_name, params) or None
        """
        message_keywords = config.get("message_keywords", ["saying", "tell them", "message", "that"])
        
        # Find message separator
        for separator in message_keywords:
            if separator in user_text:
                # Split into contact and message parts
                parts = user_text.split(separator, 1)
                if len(parts) == 2:
                    contact_part = parts[0]
                    message_part = parts[1].strip()
                    
                    # Remove command keywords from contact part
                    for keyword in config["remove_keywords"]:
                        contact_part = contact_part.replace(keyword, "")
                    contact = contact_part.strip()
                    
                    if contact and message_part:
                        logger.info(f"✅ WhatsApp: contact={contact}, message={message_part}")
                        return ("send_whatsapp", {"contact": contact, "message": message_part})
        
        return None
    
    def _extract_number(self, text: str) -> Optional[int]:
        """
        Extract number from text
        Supports both digits and word numbers (one, two, ten, etc.)
        
        Args:
            text: Text containing number
        
        Returns:
            Extracted number or None
        """
        # Try to find digits first
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
        
        # Word to number mapping
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90, "hundred": 100,
            "full": 100, "max": 100, "half": 50, "mute": 0, "low": 20
        }
        
        for word, num in word_to_num.items():
            if word in text:
                return num
        
        return None
    
    def _is_valid_domain(self, text: str) -> bool:
        """
        Check if text looks like a domain name
        
        Args:
            text: Text to validate
        
        Returns:
            True if looks like a domain
        """
        # Common TLDs
        tlds = [
            '.com', '.org', '.net', '.io', '.edu', '.gov', '.co', '.ai',
            '.app', '.dev', '.in', '.uk', '.ca', '.au', '.de', '.fr',
            '.jp', '.cn', '.info', '.biz', '.tech', '.online', '.xyz'
        ]
        
        # Check if contains TLD
        return any(tld in text for tld in tlds)
    
    def get_command_list(self) -> Dict[str, list]:
        """
        Get list of all available commands for help/documentation
        
        Returns:
            Dictionary of command categories with example commands
        """
        return {
            "time_date": [
                "What time is it?",
                "What's the date?",
                "Tell me the time"
            ],
            "system": [
                "Battery status",
                "System info",
                "How much battery?"
            ],
            "apps": [
                "Open Spotify",
                "Open WhatsApp",  # NEW
                "Launch Calculator",
                "Open Notepad",
                "Start Task Manager",
                "Open Paint"
            ],
            "web_browsing": [
                "Open Google",
                "Open YouTube",
                "Search Google for Python tutorials",
                "Open website github.com",
                "Close browser"
            ],
            "youtube": [
                "Play Sunflower on YouTube",  # ✅ Now works correctly!
                "Play Bohemian Rhapsody YouTube",
                "Search YouTube for Interstellar",
                "Pause YouTube",
                "Resume video"
            ],
            "spotify": [
                "Play Sunflower by Rex Orange County",  # ✅ Now works correctly!
                "Play Blinding Lights on Spotify",
                "Play Imagine",  # Generic play -> Spotify
                "Pause music",
                "Next song",
                "Set volume to 50",
                "What's playing?",
                "Play downloaded songs"
            ],
            "files": [
                "Open downloads",
                "Search for report.pdf",
                "Create folder Project Files",
                "Open file presentation.pptx"
            ],
            "whatsapp": [
                "Open WhatsApp",  # NEW
                "Send WhatsApp to Mom saying Hello",
                "Message John on WhatsApp saying Meeting at 5"
            ]
        }
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get command registry statistics
        
        Returns:
            Dictionary with command counts
        """
        return {
            "direct_commands": len(self.direct_commands),
            "parameterized_patterns": len(self.parameterized_patterns),
            "total_command_variants": len(self.direct_commands) + len(self.parameterized_patterns)
        }


# ============================================
# Factory Function
# ============================================

def create_command_registry(function_registry: FunctionRegistry) -> CommandRegistry:
    """
    Factory function to create command registry
    
    Args:
        function_registry: Initialized FunctionRegistry
    
    Returns:
        CommandRegistry instance
    """
    registry = CommandRegistry(function_registry)
    stats = registry.get_stats()
    
    logger.info(f"📝 Command registry initialized:")
    logger.info(f"   - Direct commands: {stats['direct_commands']}")
    logger.info(f"   - Parameterized patterns: {stats['parameterized_patterns']}")
    
    return registry
