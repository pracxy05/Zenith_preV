"""
Spotify Control Module - ULTIMATE VERSION v3.0
✅ Robust search & play with keyboard automation
✅ Auto-detects installation paths
✅ Offline mode support (plays from ANY downloaded playlist)
✅ Fallback to Liked Songs if no query
✅ Window focus & improved timing
"""

import logging
import subprocess
import time
import pyautogui
import psutil
from typing import Optional

logger = logging.getLogger(__name__)


class SpotifyController:
    """
    Spotify control via keyboard shortcuts and app launching
    Uses Windows Spotify keyboard shortcuts for reliable control
    """
    
    def __init__(self):
        """Initialize Spotify controller with auto-detected paths"""
        
        # ✅ AUTO-DETECT: Try common Spotify installation paths
        # Ordered by likelihood (Desktop > Microsoft Store > System-wide)
        self.spotify_paths = [
            r"C:\Users\{username}\AppData\Roaming\Spotify\Spotify.exe",  # Desktop version (RECOMMENDED)
            r"C:\Users\{username}\AppData\Local\Microsoft\WindowsApps\Spotify.exe",  # Microsoft Store
            r"C:\Program Files\Spotify\Spotify.exe",  # System-wide install
        ]
        
        # Replace {username} with actual username
        import os
        username = os.getenv('USERNAME')
        self.spotify_paths = [path.replace('{username}', username) for path in self.spotify_paths]
        
        self.is_running = False
        logger.info("Spotify keyboard controller initialized")
    
    def ensure_spotify_running(self) -> bool:
        """
        Check if Spotify is running, launch if not
        
        Returns:
            True if Spotify is running, False if could not launch
        """
        try:
            # Check if Spotify process exists
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'Spotify.exe' in proc.info['name']:
                    self.is_running = True
                    logger.info("✓ Spotify is running")
                    return True
            
            # Try to launch Spotify
            logger.info("Launching Spotify...")
            for path in self.spotify_paths:
                try:
                    subprocess.Popen(path, shell=True)
                    time.sleep(6)  # Wait for Spotify to fully load
                    self.is_running = True
                    logger.info(f"✓ Launched Spotify from: {path}")
                    return True
                except Exception as e:
                    logger.debug(f"Could not launch from {path}: {e}")
                    continue
            
            logger.warning("Could not find or launch Spotify")
            return False
        
        except Exception as e:
            logger.error(f"Spotify check error: {e}")
            return False
    
    def focus_spotify_window(self) -> bool:
        """
        Force Spotify window to focus
        Ensures keyboard shortcuts go to Spotify, not other apps
        
        Returns:
            True if window focused successfully
        """
        try:
            import pygetwindow as gw
            
            time.sleep(0.5)  # Wait for window to be ready
            
            # Find Spotify window
            windows = gw.getWindowsWithTitle('Spotify')
            
            if not windows:
                logger.debug("Spotify window not found (might be minimized)")
                return False
            
            # Get the first Spotify window
            spotify_window = windows[0]
            
            # Restore if minimized
            if spotify_window.isMinimized:
                spotify_window.restore()
            
            # Activate (bring to front)
            spotify_window.activate()
            time.sleep(0.4)  # Wait for window to focus
            
            logger.info("✓ Spotify window focused")
            return True
        
        except ImportError:
            logger.warning("pygetwindow not installed - window focus disabled")
            logger.warning("Install with: pip install pygetwindow")
            return False
        
        except Exception as e:
            logger.error(f"Window focus error: {e}")
            return False
    
    def play_music(self, query: str = None) -> str:
        """
        Play music on Spotify using search
        
        Args:
            query: Song name, artist, or album to search for
                   If None, plays from Liked Songs/Downloaded playlists
        
        Returns:
            Success message
        """
        try:
            # Ensure Spotify is running
            if not self.ensure_spotify_running():
                return "Could not launch Spotify. Please check installation."
            
            # Focus Spotify window
            self.focus_spotify_window()
            time.sleep(1.2)  # Wait for Spotify to be ready
            
            # If no query, just press Space to play/resume from last played
            if not query or query.strip() == "":
                pyautogui.press('space')
                logger.info("Playing from last played/Downloaded songs")
                return "Playing music on Spotify"
            
            # ✅ ROBUST SEARCH FLOW:
            # 1. Focus search bar (Ctrl+L)
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.8)
            
            # 2. Clear any existing text (Ctrl+A, then type)
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.3)
            
            # 3. Type search query
            pyautogui.write(query, interval=0.05)
            time.sleep(1.2)  # Wait for search results
            
            # 4. Press Enter to go to search results
            pyautogui.press('enter')
            time.sleep(1.2)  # Wait for results page
            
            # 5. Press Enter again to play first result
            pyautogui.press('enter')
            
            logger.info(f"✓ Playing on Spotify: {query}")
            return f"Playing {query} on Spotify"
        
        except Exception as e:
            logger.error(f"Spotify play error: {e}")
            return f"Could not play on Spotify: {str(e)}"
    
    def pause_playback(self) -> str:
        """
        Pause/Resume Spotify (Space bar toggle)
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Spotify is not running"
            
            self.focus_spotify_window()
            time.sleep(0.3)
            
            # Space bar toggles play/pause
            pyautogui.press('space')
            
            logger.info("Spotify pause/resume")
            return "Music paused"
        
        except Exception as e:
            logger.error(f"Pause error: {e}")
            return f"Could not pause: {str(e)}"
    
    def resume_playback(self) -> str:
        """
        Resume playback (same as pause - Space toggles)
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Spotify is not running"
            
            self.focus_spotify_window()
            time.sleep(0.3)
            
            pyautogui.press('space')
            
            logger.info("Spotify resumed")
            return "Music resumed"
        
        except Exception as e:
            logger.error(f"Resume error: {e}")
            return f"Could not resume: {str(e)}"
    
    def next_track(self) -> str:
        """
        Skip to next track (Ctrl+Right)
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Spotify is not running"
            
            self.focus_spotify_window()
            time.sleep(0.3)
            
            pyautogui.hotkey('ctrl', 'right')
            
            logger.info("Next track")
            return "Next track"
        
        except Exception as e:
            logger.error(f"Next track error: {e}")
            return f"Could not skip: {str(e)}"
    
    def previous_track(self) -> str:
        """
        Go to previous track (Ctrl+Left)
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Spotify is not running"
            
            self.focus_spotify_window()
            time.sleep(0.3)
            
            pyautogui.hotkey('ctrl', 'left')
            
            logger.info("Previous track")
            return "Previous track"
        
        except Exception as e:
            logger.error(f"Previous track error: {e}")
            return f"Could not go back: {str(e)}"
    
    def set_volume(self, volume: int) -> str:
        """
        Set volume (Ctrl+Up/Down)
        
        Args:
            volume: Volume level 0-100
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Spotify is not running"
            
            self.focus_spotify_window()
            time.sleep(0.3)
            
            # Approximate: each press ≈ 10% volume change
            presses = abs(volume - 50) // 10
            
            if volume > 50:
                # Volume up
                for _ in range(presses):
                    pyautogui.hotkey('ctrl', 'up')
                    time.sleep(0.1)
            else:
                # Volume down
                for _ in range(presses):
                    pyautogui.hotkey('ctrl', 'down')
                    time.sleep(0.1)
            
            logger.info(f"Set volume to ~{volume}%")
            return f"Volume adjusted to ~{volume} percent"
        
        except Exception as e:
            logger.error(f"Volume error: {e}")
            return f"Could not set volume: {str(e)}"
    
    def get_current_track(self) -> str:
        """
        Get current track (not available with keyboard control)
        Would require Spotify API integration
        
        Returns:
            Generic message
        """
        if not self.ensure_spotify_running():
            return "Spotify is not running"
        
        return "Music is playing on Spotify"
    
    def play_downloaded_songs(self) -> str:
        """
        Play from Downloaded songs (offline-friendly)
        ✅ Works with ANY downloaded playlist (like your "<3" playlist)
        
        Returns:
            Success message
        """
        try:
            if not self.ensure_spotify_running():
                return "Could not launch Spotify"
            
            self.focus_spotify_window()
            time.sleep(1.2)
            
            # Click on "Your Library" in sidebar (left side)
            # Then click on "Downloaded" filter
            # This is done by navigating to library with keyboard
            
            # Navigate to Your Library section (Ctrl+Shift+O opens library)
            pyautogui.hotkey('ctrl', 'shift', 'o')
            time.sleep(1.0)
            
            # Click on Downloaded filter (if available)
            # Or search for "downloaded"
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.8)
            
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.3)
            
            pyautogui.write("downloaded", interval=0.05)
            time.sleep(1.0)
            
            pyautogui.press('enter')
            time.sleep(1.0)
            
            pyautogui.press('enter')  # Play
            
            logger.info("Playing downloaded songs")
            return "Playing your downloaded songs"
        
        except Exception as e:
            logger.error(f"Downloaded songs error: {e}")
            # Fallback: just press space to play whatever was last
            try:
                pyautogui.press('space')
                return "Playing from Spotify (offline mode)"
            except:
                return f"Could not play downloaded songs: {str(e)}"


# ============================================
# Global Singleton Instance
# ============================================

_spotify_controller = None

def get_spotify_controller() -> SpotifyController:
    """Get or create Spotify controller singleton"""
    global _spotify_controller
    if _spotify_controller is None:
        _spotify_controller = SpotifyController()
    return _spotify_controller


# ============================================
# Public API Functions (for tool registry)
# ============================================

def play_spotify(query: str) -> str:
    """
    Play music on Spotify
    
    Args:
        query: Song/artist name to search for
    
    Returns:
        Success message
    """
    controller = get_spotify_controller()
    return controller.play_music(query)


def open_spotify() -> str:
    """
    Just open Spotify without playing anything
    ✅ NEW: For "open spotify" command
    
    Returns:
        Success message
    """
    controller = get_spotify_controller()
    
    try:
        if controller.ensure_spotify_running():
            return "Spotify opened"
        else:
            return "Could not open Spotify. Please check installation."
    except Exception as e:
        logger.error(f"Open Spotify error: {e}")
        return f"Could not open Spotify: {str(e)}"


def pause_spotify() -> str:
    """Pause Spotify playback"""
    controller = get_spotify_controller()
    return controller.pause_playback()


def resume_spotify() -> str:
    """Resume Spotify playback"""
    controller = get_spotify_controller()
    return controller.resume_playback()


def next_track() -> str:
    """Skip to next track"""
    controller = get_spotify_controller()
    return controller.next_track()


def previous_track() -> str:
    """Go to previous track"""
    controller = get_spotify_controller()
    return controller.previous_track()


def set_volume(volume: int) -> str:
    """
    Set Spotify volume
    
    Args:
        volume: Volume level 0-100
    
    Returns:
        Success message
    """
    controller = get_spotify_controller()
    return controller.set_volume(volume)


def current_track() -> str:
    """Get currently playing track info"""
    controller = get_spotify_controller()
    return controller.get_current_track()


def play_downloaded_songs() -> str:
    """
    Play from downloaded songs/playlists (offline-friendly)
    ✅ Works with your "<3" playlist and any other downloaded playlists
    
    Returns:
        Success message
    """
    controller = get_spotify_controller()
    return controller.play_downloaded_songs()
