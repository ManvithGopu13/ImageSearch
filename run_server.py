"""
Server Startup Script
=====================
Convenient script to start the FastAPI server with proper configuration.

Usage:
    python run_server.py
    
    or with custom host/port:
    python run_server.py --host 0.0.0.0 --port 8080
"""

import uvicorn
import argparse
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import settings


def main():
    """
    Start the FastAPI server.
    
    Learning Points:
    - Uvicorn is an ASGI server for async Python web apps
    - --reload enables auto-reload on code changes (development)
    - workers > 1 for production (handles concurrent requests)
    """
    parser = argparse.ArgumentParser(description='Start the Image Search RAG API Server')
    parser.add_argument('--host', type=str, default=settings.api_host,
                       help=f'Host to bind to (default: {settings.api_host})')
    parser.add_argument('--port', type=int, default=settings.api_port,
                       help=f'Port to bind to (default: {settings.api_port})')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload on code changes (development)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (production)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 Starting Image Search RAG System")
    print("="*70)
    print(f"📡 API Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🌐 Frontend: Open frontend/index.html in your browser")
    print(f"💾 Database: {settings.chroma_persist_directory}")
    print(f"📁 Uploads: {settings.upload_dir}")
    print(f"🖥️  Device: {settings.device}")
    print("="*70)
    print()
    
    # Start server
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # Workers incompatible with reload
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()

