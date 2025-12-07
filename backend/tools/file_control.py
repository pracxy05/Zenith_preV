"""
File Control Module
Handles file explorer, file operations, and folder management
"""

import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FileController:
    """File system operations and explorer control"""
    
    def __init__(self):
        """Initialize file controller"""
        self.common_folders = {
            "downloads": str(Path.home() / "Downloads"),
            "documents": str(Path.home() / "Documents"),
            "desktop": str(Path.home() / "Desktop"),
            "pictures": str(Path.home() / "Pictures"),
            "music": str(Path.home() / "Music"),
            "videos": str(Path.home() / "Videos"),
        }
        logger.info("File controller initialized")
    
    def open_file_explorer(self, location: str = None) -> str:
        """
        Open Windows File Explorer
        
        Args:
            location: Folder name or path (e.g., "downloads", "documents")
            
        Returns:
            Success message
        """
        try:
            # Default to user home directory
            if not location:
                path = str(Path.home())
            # Check common folders
            elif location.lower() in self.common_folders:
                path = self.common_folders[location.lower()]
            # Custom path
            else:
                path = location
            
            # Verify path exists
            if not os.path.exists(path):
                return f"Folder not found: {location}"
            
            # Open explorer
            subprocess.Popen(f'explorer "{path}"')
            
            logger.info(f"Opened file explorer: {path}")
            return f"Opened {os.path.basename(path)}"
            
        except Exception as e:
            logger.error(f"Open explorer error: {e}")
            return f"Could not open file explorer: {str(e)}"
    
    def search_files(self, filename: str, directory: str = None) -> str:
        """
        Search for files by name
        
        Args:
            filename: File name or pattern to search
            directory: Directory to search in (default: user home)
            
        Returns:
            List of found files
        """
        try:
            search_dir = directory if directory else str(Path.home())
            
            if not os.path.exists(search_dir):
                return f"Directory not found: {directory}"
            
            # Search for files
            found_files = []
            filename_lower = filename.lower()
            
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if filename_lower in file.lower():
                        found_files.append(os.path.join(root, file))
                        if len(found_files) >= 5:  # Limit to 5 results
                            break
                if len(found_files) >= 5:
                    break
            
            if not found_files:
                return f"No files found matching: {filename}"
            
            result = f"Found {len(found_files)} files:\n"
            for file_path in found_files:
                result += f"- {os.path.basename(file_path)}\n"
            
            logger.info(f"Found {len(found_files)} files matching {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Search files error: {e}")
            return f"Error searching: {str(e)}"
    
    def open_file(self, filename: str) -> str:
        """
        Open a file with default application
        
        Args:
            filename: File name or full path
            
        Returns:
            Success message
        """
        try:
            # If not a full path, search in common locations
            if not os.path.isabs(filename):
                # Search in common folders
                for folder_path in self.common_folders.values():
                    potential_path = os.path.join(folder_path, filename)
                    if os.path.exists(potential_path):
                        filename = potential_path
                        break
            
            if not os.path.exists(filename):
                return f"File not found: {filename}"
            
            # Open file with default application
            os.startfile(filename)
            
            logger.info(f"Opened file: {filename}")
            return f"Opened {os.path.basename(filename)}"
            
        except Exception as e:
            logger.error(f"Open file error: {e}")
            return f"Could not open file: {str(e)}"
    
    def create_folder(self, folder_name: str, location: str = None) -> str:
        """
        Create a new folder
        
        Args:
            folder_name: Name of folder to create
            location: Parent directory (default: Desktop)
            
        Returns:
            Success message
        """
        try:
            # Default to Desktop
            if not location:
                location = self.common_folders["desktop"]
            elif location.lower() in self.common_folders:
                location = self.common_folders[location.lower()]
            
            # Create full path
            folder_path = os.path.join(location, folder_name)
            
            if os.path.exists(folder_path):
                return f"Folder already exists: {folder_name}"
            
            # Create folder
            os.makedirs(folder_path)
            
            logger.info(f"Created folder: {folder_path}")
            return f"Created folder: {folder_name}"
            
        except Exception as e:
            logger.error(f"Create folder error: {e}")
            return f"Could not create folder: {str(e)}"
    
    def delete_file(self, filename: str) -> str:
        """
        Move file to recycle bin
        
        Args:
            filename: File name or path
            
        Returns:
            Success message
        """
        try:
            if not os.path.exists(filename):
                return f"File not found: {filename}"
            
            # Move to recycle bin (Windows)
            from win32com.shell import shell, shellcon
            shell.SHFileOperation((
                0,
                shellcon.FO_DELETE,
                filename,
                None,
                shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION,
                None,
                None
            ))
            
            logger.info(f"Deleted file: {filename}")
            return f"Deleted {os.path.basename(filename)}"
            
        except ImportError:
            logger.error("pywin32 not installed")
            return "Cannot delete: pywin32 not installed"
        except Exception as e:
            logger.error(f"Delete file error: {e}")
            return f"Could not delete: {str(e)}"


# Global instance
_file_controller = None


def get_file_controller() -> FileController:
    """Get or create file controller singleton"""
    global _file_controller
    if _file_controller is None:
        _file_controller = FileController()
    return _file_controller


# ============================================
# Public Functions for Function Registry
# ============================================

def open_file_explorer(location: str = None) -> str:
    """
    Open file explorer
    
    Args:
        location: Folder name (downloads, documents, desktop, etc.)
    """
    controller = get_file_controller()
    return controller.open_file_explorer(location)


def search_files(filename: str, directory: str = None) -> str:
    """
    Search for files
    
    Args:
        filename: File name to search for
        directory: Directory to search in (optional)
    """
    controller = get_file_controller()
    return controller.search_files(filename, directory)


def open_file(filename: str) -> str:
    """
    Open a file
    
    Args:
        filename: File name or path
    """
    controller = get_file_controller()
    return controller.open_file(filename)


def create_folder(folder_name: str, location: str = None) -> str:
    """
    Create a new folder
    
    Args:
        folder_name: Name of folder
        location: Parent directory (default: Desktop)
    """
    controller = get_file_controller()
    return controller.create_folder(folder_name, location)


def delete_file(filename: str) -> str:
    """
    Delete a file (move to recycle bin)
    
    Args:
        filename: File name or path
    """
    controller = get_file_controller()
    return controller.delete_file(filename)
