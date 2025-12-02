#!/usr/bin/env python
"""
Startup script for Advanced Reorder Prediction System
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'tensorflow',
        'lightgbm',
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'sklearn'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n Please install them with:")
        print(" pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed")
    return True


def start_server():
    """Start the FastAPI server"""
    print("\n Starting API server...")
    print("   API will be available at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")

    # Start uvicorn
    subprocess.Popen([
        sys.executable, '-m', 'uvicorn',
        'app.main:app',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--reload'
    ])

    print("\n✅ Server started successfully")


def start_frontend():
    """Start a simple HTTP server for the frontend"""
    print("\n Starting frontend server...")
    print("   Frontend will be available at: http://localhost:8080")

    os.chdir('app/static')
    subprocess.Popen([
        sys.executable, '-m', 'http.server', '8080'
    ])

    print("\n✅ Frontend server started")


def main():
    """Main startup routine"""
    print("=" * 60)
    print(" Advanced Reorder Prediction System v2.0")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create necessary directories
    os.makedirs('models_store', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    print("✅ Directories created")

    # Start servers
    start_server()

    # Small delay to let the API server start
    import time
    time.sleep(3)

    start_frontend()

    # Open browser
    print("\n Opening browser...")
    time.sleep(2)
    webbrowser.open('http://localhost:8080')

    print("\n" + "=" * 60)
    print("✅ System is ready!")
    print("=" * 60)
    print("\n Quick Start Guide:")
    print("   1. Upload your CSV data file")
    print("   2. Click 'Train All Models'")
    print("   3. Enter a customer ID to get predictions")
    print("\n Press Ctrl+C to stop all servers")
    print("=" * 60)

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n Shutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()