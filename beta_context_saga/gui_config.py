"""
Configuration module for the ContextSaga GUI application.
This module provides functions for managing GUI settings and database paths.
"""

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".context_saga" / "config.json"
DEFAULT_CONFIG = {
    "memory_db_path": "data/memory.db",
    "chroma_db_path": "data/chroma_db",
    "port": 5000,
    "host": "0.0.0.0",
    "debug": True,
}


def ensure_config_dir_exists(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Ensure the configuration directory exists."""
    config_path.parent.mkdir(parents=True, exist_ok=True)


def get_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Get the configuration settings.
    If the configuration file doesn't exist, create it with default values.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration settings
    """
    ensure_config_dir_exists(config_path)

    if not config_path.exists():
        # Create default config file
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG.copy()

    # Read existing config
    try:
        with open(config_path) as f:
            config = json.load(f)

        # Update config with any missing default values
        updated = False
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
                updated = True

        # Save updated config if needed
        if updated:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

        return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        return DEFAULT_CONFIG.copy()


def update_config(
    updates: dict[str, Any], config_path: Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    """
    Update the configuration with new values.

    Args:
        updates: Dictionary containing the updates to apply
        config_path: Path to the configuration file

    Returns:
        The updated configuration dictionary
    """
    config = get_config(config_path)

    # Apply updates
    for key, value in updates.items():
        config[key] = value

    # Save updated config
    ensure_config_dir_exists(config_path)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config


def get_db_paths() -> tuple[str, str]:
    """
    Get the paths for the memory and Chroma databases.

    Returns:
        Tuple containing (memory_db_path, chroma_db_path)
    """
    config = get_config()
    memory_db_path = config.get("memory_db_path", DEFAULT_CONFIG["memory_db_path"])
    chroma_db_path = config.get("chroma_db_path", DEFAULT_CONFIG["chroma_db_path"])

    # Convert to absolute paths if needed
    if not os.path.isabs(memory_db_path):
        memory_db_path = os.path.abspath(memory_db_path)
    if not os.path.isabs(chroma_db_path):
        chroma_db_path = os.path.abspath(chroma_db_path)

    # Ensure directories exist
    os.makedirs(os.path.dirname(memory_db_path), exist_ok=True)
    os.makedirs(chroma_db_path, exist_ok=True)

    return memory_db_path, chroma_db_path


def get_server_settings() -> tuple[str, int, bool]:
    """
    Get the server host, port, and debug settings.

    Returns:
        Tuple containing (host, port, debug)
    """
    config = get_config()
    host = config.get("host", DEFAULT_CONFIG["host"])
    port = int(config.get("port", DEFAULT_CONFIG["port"]))
    debug = bool(config.get("debug", DEFAULT_CONFIG["debug"]))

    return host, port, debug
