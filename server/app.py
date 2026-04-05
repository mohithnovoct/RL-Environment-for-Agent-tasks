# Copyright (c) 2026. All rights reserved.
# Data Cleaning Environment - FastAPI Application

"""
FastAPI application for the Data Cleaning Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

import sys
import os

# Ensure parent directory is on path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core import create_app

from server.data_cleaning_environment import DataCleaningEnvironment
from models import DataCleaningAction, DataCleaningObservation

# Create the app with the environment class (factory pattern for WebSocket sessions)
app = create_app(
    DataCleaningEnvironment,
    DataCleaningAction,
    DataCleaningObservation,
    env_name="data_cleaning_env",
)

try:
    import gradio as gr
    from app import demo
    # Mount Gradio frontend UI to the FastAPI app at root
    # Note: FastAPI resolves endpoints first, so /reset and /step keep working perfectly.
    app = gr.mount_gradio_app(app, demo, path="/")
except Exception as e:
    print(f"Failed to mount Gradio UI: {e}")


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
