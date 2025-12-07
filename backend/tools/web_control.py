"""
Web Control Module - Chrome Automation
Handles web browsing, search, and YouTube control
Version 4.4 - With Chrome Profile Support

IMPORTANT: This uses YOUR Chrome profile so you stay logged in to:
- Google Account
- YouTube (subscriptions, history)
- WhatsApp Web
- All other websites
"""

import time
import logging
import subprocess
import os
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

# ============================================
# 🔧 CHROME PROFILE CONFIGURATION
# ============================================
# Change these paths if you want to use a different Chrome profile
#
# To find your profile:
# 1. Open Chrome and go to: chrome://version
# 2. Look for "Profile Path" - it shows your current profile
# 3. User Data Dir = everything before the profile folder name
# 4. Profile Directory = the folder name (Default, Profile 1, Profile 2, etc.)
#
# Example paths:
#   User Data Dir: C:\Users\praharsh\AppData\Local\Google\Chrome\User Data
#   Profile Dir:   Profile 1
# ============================================

CHROME_USER_DATA_DIR = r"C:\Users\praharsh\AppData\Local\Google\Chrome\User Data"
CHROME_PROFILE_DIR = "Profile 2"  # Your main profile (13GB one)

# ============================================
# 🔧 WHATSAPP CHROME APP PATH
# ============================================
# This is the shortcut to WhatsApp Web Chrome App
# Change this if your WhatsApp shortcut is in a different location
# ============================================

WHATSAPP_SHORTCUT_PATH = r"C:\Users\praharsh\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Chrome Apps\WhatsApp Web.lnk"


class WebController:
    """
    Web browser automation for Chrome
    Handles web search, YouTube, and general browsing
    Uses YOUR Chrome profile to maintain login sessions
    """
    
    def __init__(self):
        """Initialize web controller"""
        self.driver: Optional[webdriver.Chrome] = None
        self.is_browser_open = False
        logger.info("Web controller initialized")
    
    def _ensure_browser(self) -> webdriver.Chrome:
        """
        Ensure browser is open and return driver
        
        Returns:
            Chrome WebDriver instance with your profile loaded
        """
        if self.driver is None or not self.is_browser_open:
            try:
                # ============================================
                # Chrome options for better performance
                # ============================================
                options = Options()
                
                # Start maximized
                options.add_argument('--start-maximized')
                
                # ============================================
                # 🔑 CRITICAL: Load YOUR Chrome profile
                # ============================================
                # This makes Chrome use your existing profile with:
                # - Saved passwords
                # - Logged-in sessions (Google, YouTube, WhatsApp)
                # - Bookmarks
                # - Extensions
                # ============================================
                options.add_argument(f'--user-data-dir={CHROME_USER_DATA_DIR}')
                options.add_argument(f'--profile-directory={CHROME_PROFILE_DIR}')
                
                # ============================================
                # Anti-detection options (makes automation smoother)
                # ============================================
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                
                # ============================================
                # Additional stability options
                # ============================================
                options.add_argument('--no-first-run')
                options.add_argument('--no-default-browser-check')
                options.add_argument('--disable-popup-blocking')
                
                # ============================================
                # IMPORTANT: Close any existing Chrome windows first
                # Selenium can't attach to existing Chrome, so we need
                # exclusive access to the profile
                # ============================================
                # Uncomment below if you want auto-close (risky if you have unsaved work)
                # os.system("taskkill /f /im chrome.exe >nul 2>&1")
                # time.sleep(1)
                
                # Initialize Chrome driver
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                self.is_browser_open = True
                
                logger.info(f"Chrome browser opened with profile: {CHROME_PROFILE_DIR}")
                return self.driver
                
            except Exception as e:
                logger.error(f"Failed to open browser: {e}")
                
                # ============================================
                # Common error: Chrome profile is already in use
                # ============================================
                if "user data directory is already in use" in str(e).lower():
                    logger.error("Chrome profile is already open! Close Chrome and try again.")
                    raise RuntimeError(
                        "Chrome profile is already in use. Please close all Chrome windows first, "
                        "or use a different profile in CHROME_PROFILE_DIR setting."
                    )
                raise RuntimeError(f"Could not open Chrome: {e}")
        
        return self.driver
    
    def open_website(self, url: str) -> str:
        """
        Open a website in Chrome
        
        Args:
            url: Website URL (can be without https://)
        
        Returns:
            Success message
        """
        try:
            # Add https:// if not present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            driver = self._ensure_browser()
            driver.get(url)
            
            # Wait for page to load
            time.sleep(2)
            
            logger.info(f"Opened website: {url}")
            return f"Opened {url}"
            
        except Exception as e:
            logger.error(f"Failed to open website: {e}")
            return f"Error opening website: {str(e)}"
    
    def open_whatsapp(self) -> str:
        """
        Open WhatsApp Web using Chrome App shortcut
        This opens WhatsApp in your logged-in profile
        
        Returns:
            Success message
        """
        try:
            # ============================================
            # Method 1: Use Chrome App shortcut (preferred)
            # This opens WhatsApp as a standalone app window
            # ============================================
            if os.path.exists(WHATSAPP_SHORTCUT_PATH):
                subprocess.Popen(['start', '', WHATSAPP_SHORTCUT_PATH], shell=True)
                logger.info("WhatsApp Web opened via Chrome App")
                return "WhatsApp opened"
            
            # ============================================
            # Method 2: Fallback to browser
            # Opens WhatsApp Web in Chrome browser
            # ============================================
            logger.info("WhatsApp shortcut not found, opening in browser...")
            return self.open_website("web.whatsapp.com")
            
        except Exception as e:
            logger.error(f"Failed to open WhatsApp: {e}")
            return f"Error opening WhatsApp: {str(e)}"
    
    def search_google(self, query: str) -> str:
        """
        Search Google and open results
        
        Args:
            query: Search query
        
        Returns:
            Success message
        """
        try:
            driver = self._ensure_browser()
            
            # Open Google
            driver.get("https://www.google.com")
            time.sleep(1)
            
            # Find search box
            search_box = driver.find_element(By.NAME, "q")
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            # Wait for results
            time.sleep(2)
            
            logger.info(f"Google search: {query}")
            return f"Searched for: {query}"
            
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return f"Error searching: {str(e)}"
    
    def search_youtube(self, query: str) -> str:
        """
        Search YouTube
        
        Args:
            query: Search query
        
        Returns:
            Success message
        """
        try:
            driver = self._ensure_browser()
            
            # Open YouTube
            driver.get("https://www.youtube.com")
            time.sleep(2)
            
            # Find search box
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "search_query"))
                )
                search_box.clear()
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                
                # Wait for results
                time.sleep(3)
                
                logger.info(f"YouTube search: {query}")
                return f"Searched YouTube for: {query}"
                
            except Exception as e:
                logger.error(f"YouTube search box not found: {e}")
                return "Could not find YouTube search box"
                
        except Exception as e:
            logger.error(f"YouTube search failed: {e}")
            return f"Error searching YouTube: {str(e)}"
    
    def play_youtube(self, query: str) -> str:
        """
        Search and play first YouTube video
        
        Args:
            query: Search query for video
        
        Returns:
            Success message
        """
        try:
            driver = self._ensure_browser()
            
            # Open YouTube
            driver.get("https://www.youtube.com")
            time.sleep(2)
            
            # Search for video
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "search_query"))
                )
                search_box.clear()
                search_box.send_keys(query)
                search_box.send_keys(Keys.RETURN)
                
                # Wait for results and click first video
                time.sleep(3)
                
                # Find first video result
                video = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "ytd-video-renderer #video-title"))
                )
                video.click()
                
                # Wait for video to start
                time.sleep(2)
                
                logger.info(f"Playing YouTube video: {query}")
                return f"Playing: {query}"
                
            except Exception as e:
                logger.error(f"Could not play video: {e}")
                return f"Could not find video: {query}"
                
        except Exception as e:
            logger.error(f"YouTube play failed: {e}")
            return f"Error playing video: {str(e)}"
    
    def youtube_pause(self) -> str:
        """
        Pause/Resume YouTube video (press K key)
        
        Returns:
            Success message
        """
        try:
            if not self.is_browser_open or self.driver is None:
                return "No browser is open"
            
            # Press 'k' key to pause/play YouTube
            actions = ActionChains(self.driver)
            actions.send_keys('k')
            actions.perform()
            
            logger.info("YouTube pause/play toggled")
            return "Video paused or resumed"
            
        except Exception as e:
            logger.error(f"YouTube pause failed: {e}")
            return f"Error pausing video: {str(e)}"
    
    def close_browser(self) -> str:
        """
        Close Chrome browser
        
        Returns:
            Success message
        """
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                self.is_browser_open = False
                logger.info("Chrome browser closed")
                return "Browser closed"
            else:
                return "Browser is not open"
                
        except Exception as e:
            logger.error(f"Failed to close browser: {e}")
            return f"Error closing browser: {str(e)}"
    
    def get_current_url(self) -> str:
        """
        Get current page URL
        
        Returns:
            Current URL or error message
        """
        try:
            if not self.is_browser_open or self.driver is None:
                return "No browser is open"
            
            url = self.driver.current_url
            logger.info(f"Current URL: {url}")
            return f"Current page: {url}"
            
        except Exception as e:
            logger.error(f"Error getting URL: {e}")
            return f"Error getting URL: {str(e)}"


# Global instance
_web_controller = None

def get_web_controller() -> WebController:
    """Get or create web controller singleton"""
    global _web_controller
    if _web_controller is None:
        _web_controller = WebController()
    return _web_controller


# ============================================
# Public Functions for Function Registry
# ============================================

def open_website(url: str) -> str:
    """
    Open a website in Chrome
    
    Args:
        url: Website URL (e.g., "google.com" or "https://youtube.com")
    """
    controller = get_web_controller()
    return controller.open_website(url)

def open_whatsapp() -> str:
    """
    Open WhatsApp Web (uses Chrome App shortcut or browser)
    Opens in your logged-in profile
    """
    controller = get_web_controller()
    return controller.open_whatsapp()

def search_google(query: str) -> str:
    """
    Search Google
    
    Args:
        query: Search query
    """
    controller = get_web_controller()
    return controller.search_google(query)

def search_youtube(query: str) -> str:
    """
    Search YouTube
    
    Args:
        query: Search query
    """
    controller = get_web_controller()
    return controller.search_youtube(query)

def play_youtube(query: str) -> str:
    """
    Search and play YouTube video
    
    Args:
        query: Video search query
    """
    controller = get_web_controller()
    return controller.play_youtube(query)

def youtube_pause() -> str:
    """Pause or resume YouTube video"""
    controller = get_web_controller()
    return controller.youtube_pause()

def close_browser() -> str:
    """Close Chrome browser"""
    controller = get_web_controller()
    return controller.close_browser()

def get_current_url() -> str:
    """Get current page URL"""
    controller = get_web_controller()
    return controller.get_current_url()
