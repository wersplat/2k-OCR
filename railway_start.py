#!/usr/bin/env python3
"""
Railway Startup Script for NBA 2K OCR System
"""

import os
import subprocess
import time
import signal
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    dirs = [
        "processed/json",
        "processed/images", 
        "toProcess/images",
        "labelstudio_tasks",
        "labelstudio_data",
        "labelstudio_media"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def start_labelstudio():
    """Start Label Studio in background"""
    print("🏷️ Starting Label Studio...")
    
    try:
        # Start Label Studio in background
        process = subprocess.Popen([
            "label-studio", "start",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--no-browser"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for Label Studio to start
        time.sleep(10)
        
        if process.poll() is None:
            print("✅ Label Studio started successfully")
            return process
        else:
            print("❌ Label Studio failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Label Studio: {e}")
        return None

def start_dashboard():
    """Start the web dashboard"""
    print("🌐 Starting web dashboard...")
    
    try:
        # Start dashboard with uvicorn
        subprocess.run([
            "python3", "-m", "uvicorn",
            "dashboard.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload", "False"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down...")
    sys.exit(0)

def main():
    """Main startup function"""
    print("🚀 NBA 2K OCR System - Railway Startup")
    print("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create directories
    create_directories()
    
    # Start Label Studio in background
    labelstudio_process = start_labelstudio()
    
    if labelstudio_process:
        print("🎉 All services started successfully!")
        print("📊 Dashboard: http://localhost:8000")
        print("🏷️ Label Studio: http://localhost:8080")
        
        # Start dashboard (this will block)
        start_dashboard()
    else:
        print("❌ Failed to start services")
        sys.exit(1)

if __name__ == "__main__":
    main() 