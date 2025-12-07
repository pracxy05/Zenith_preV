"""
Zenith Assistant - Main Entry Point
"""

import sys
import logging
from core.assistant import ZenithAssistant

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('zenith.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """Main entry point"""
    try:
        # Initialize assistant
        assistant = ZenithAssistant()
        
        # Start listening
        assistant.listen()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.exception("Fatal error")
        sys.exit(1)

if __name__ == "__main__":
    main()
