#!/usr/bin/env python3
"""
Entry point for running the ContextSaga dashboard.
This module allows users to start the web dashboard by running:
python -m beta_context_saga.dashboard
"""

import argparse
import sys
from pathlib import Path

from beta_context_saga.gui_config import get_server_settings


def main():
    """Main entry point for running the dashboard."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Start the ContextSaga dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", help="Host address to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Update settings if provided via command line
    if any([args.host, args.port, args.debug is not None]):
        settings = {}
        if args.host:
            settings["host"] = args.host
        if args.port:
            settings["port"] = args.port
        if args.debug is not None:
            settings["debug"] = args.debug

    # Import and run the dashboard app
    try:
        # Add saga_dashboard to sys.path
        root_dir = Path(__file__).parent.parent
        sys.path.append(str(root_dir))

        from saga_dashboard.app import app

        # Get server settings
        host, port, debug = get_server_settings()

        print(f"Starting ContextSaga Dashboard on http://{host}:{port}")
        print("Press Ctrl+C to stop the server")

        # Run the app
        app.run(host=host, port=port, debug=debug)

    except ImportError as e:
        print(f"Error importing dashboard components: {e}")
        print("Make sure you have installed the development dependencies:")
        print('pip install -e ".[dev]"')
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)


# This will run when module is executed using 'python -m beta_context_saga.dashboard'
if __name__ == "__main__":
    main()
