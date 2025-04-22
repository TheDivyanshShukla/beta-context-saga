"""
Flask web application for ContextSaga dashboard.
This module provides a web-based dashboard for the ContextSaga memory management system.
"""

import datetime
import json

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from beta_context_saga.gui_config import get_db_paths, get_server_settings, update_config
from beta_context_saga.memory_db import MemoryDatabase
from beta_context_saga.retriever import Retriever


# Custom JSON encoder to handle numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Get database paths from configuration
memory_db_path, chroma_db_path = get_db_paths()

# Initialize the Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
app.json_encoder = NumpyJSONEncoder

# Create a single instance of Retriever with the configured chroma_db_path
retriever = Retriever(persist_dir=chroma_db_path)

# Create a single instance of MemoryDatabase with the configured memory_db_path
# The class now handles thread safety internally
db = MemoryDatabase(db_path=memory_db_path, retriever=retriever)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/memories")
def memories_page():
    return render_template("memories.html")


@app.route("/tags")
def tags_page():
    return render_template("tags.html")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.route("/search")
def search_page():
    return render_template("search.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


@app.route("/api/memories", methods=["GET"])
def get_memories():
    """Get all memories, optionally filtered by tags."""
    tag = request.args.get("tag")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    if tag:
        memories = db.search_by_tags([tag], require_all=True, limit=limit)
    else:
        memories = db.list_memories(limit=limit, offset=offset)

    # Convert to JSON-serializable format
    result = []
    for memory in memories:
        result.append(
            {
                "id": memory.id,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "source": memory.source,
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags,
                "metadata": memory.metadata,
                "expires_in": str(memory.expires_in) if memory.expires_in else None,
            }
        )

    return jsonify(result)


@app.route("/api/memory/<int:memory_id>", methods=["GET"])
def get_memory(memory_id):
    """Get a specific memory by ID."""
    memory = db.get_memory(memory_id)

    if not memory:
        return jsonify({"error": "Memory not found"}), 404

    result = {
        "id": memory.id,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        "source": memory.source,
        "content": memory.content,
        "importance": memory.importance,
        "tags": memory.tags,
        "metadata": memory.metadata,
        "expires_in": str(memory.expires_in) if memory.expires_in else None,
    }

    return jsonify(result)


@app.route("/api/tags", methods=["GET"])
def get_tags():
    """Get all tags with their counts."""
    tags = db.list_all_tags()

    result = []
    for tag_name, count in tags:
        result.append({"name": tag_name, "count": count})

    return jsonify(result)


@app.route("/api/tags/<tag>/memories", methods=["GET"])
def get_memories_by_tag(tag):
    """Get memories for a specific tag."""
    limit = int(request.args.get("limit", 100))
    memories = db.search_by_tags([tag], require_all=True, limit=limit)

    # Convert to JSON-serializable format
    result = []
    for memory in memories:
        result.append(
            {
                "id": memory.id,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "source": memory.source,
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags,
                "metadata": memory.metadata,
                "expires_in": str(memory.expires_in) if memory.expires_in else None,
            }
        )

    return jsonify(result)


@app.route("/api/search", methods=["GET"])
def search_memories():
    """Search memories by text query."""
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", 10))

    if not query:
        return jsonify([])

    results = db.search_by_text(query, limit=limit)

    # Convert to JSON-serializable format
    result = []
    for memory, score in results:
        result.append(
            {
                "id": memory.id,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "source": memory.source,
                "content": memory.content,
                "importance": memory.importance,
                "tags": memory.tags,
                "metadata": memory.metadata,
                "score": float(score),
                "expires_in": str(memory.expires_in) if memory.expires_in else None,
            }
        )

    return jsonify(result)


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Get analytics data about the memory database."""
    all_memories = db.list_memories(limit=1000)
    all_tags = db.list_all_tags()

    # Calculate memory stats
    total_memories = len(all_memories)

    # Count by source
    sources = {}
    for memory in all_memories:
        source = memory.source
        sources[source] = sources.get(source, 0) + 1

    # Get most common tags
    top_tags = sorted(all_tags, key=lambda x: x[1], reverse=True)[:10]

    # Calculate memory age distribution
    now = datetime.datetime.now(datetime.UTC)
    age_ranges = {"last_day": 0, "last_week": 0, "last_month": 0, "older": 0}

    for memory in all_memories:
        age = now - memory.created_at
        if age < datetime.timedelta(days=1):
            age_ranges["last_day"] += 1
        elif age < datetime.timedelta(days=7):
            age_ranges["last_week"] += 1
        elif age < datetime.timedelta(days=30):
            age_ranges["last_month"] += 1
        else:
            age_ranges["older"] += 1

    return jsonify(
        {
            "total_memories": total_memories,
            "sources": sources,
            "top_tags": [{"name": name, "count": count} for name, count in top_tags],
            "age_distribution": age_ranges,
        }
    )


@app.route("/api/memory/<int:memory_id>", methods=["DELETE"])
def delete_memory(memory_id):
    """Delete a memory."""
    success = db.delete_memory(memory_id)

    if not success:
        return jsonify({"error": "Memory not found"}), 404

    return jsonify({"success": True})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Get current settings."""
    memory_db_path, chroma_db_path = get_db_paths()
    host, port, debug = get_server_settings()

    return jsonify(
        {
            "memory_db_path": memory_db_path,
            "chroma_db_path": chroma_db_path,
            "host": host,
            "port": port,
            "debug": debug,
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update settings."""
    data = request.json

    # Validate required fields
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Only allow updating specific fields
    valid_fields = ["memory_db_path", "chroma_db_path", "host", "port", "debug"]
    updates = {k: v for k, v in data.items() if k in valid_fields}

    # Update configuration
    try:
        updated_config = update_config(updates)

        # Return success message with restart required notice
        return jsonify(
            {
                "success": True,
                "message": "Settings updated. Restart the server for changes to take effect.",
                "config": updated_config,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to update settings: {str(e)}"}), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    # Get server settings from configuration
    host, port, debug = get_server_settings()
    app.run(debug=debug, host=host, port=port)
