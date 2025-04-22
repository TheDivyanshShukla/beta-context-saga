#!/usr/bin/env python3
"""
Run the ContextSaga dashboard application.
"""

import sys
from pathlib import Path

# Add parent directory to path so modules can be imported
sys.path.append(str(Path(__file__).parent))

# Import required modules
from beta_context_saga.gui_config import get_server_settings

from .app import app

if __name__ == "__main__":
    # Get server settings from configuration
    host, port, debug = get_server_settings()

    print(f"Starting ContextSaga Dashboard on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")

    # Run the app
    app.run(host=host, port=port, debug=debug)
