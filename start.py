#!/usr/bin/env python3
"""
NBA 2K OCR System Startup Script
Provides easy access to all system components
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import easyocr
        import torch
        import ultralytics
        print("✅ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "processed/images",
        "processed/json", 
        "toProcess/images",
        "labelstudio_tasks",
        "yolo/data/images",
        "yolo/data/labels",
        "yolo/models/2k_stats_detector/weights"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")

def start_dashboard():
    """Start the web dashboard"""
    print("🚀 Starting NBA 2K OCR Dashboard...")
    
    try:
        # Change to dashboard directory
        os.chdir("dashboard")
        
        # Start the dashboard
        subprocess.run(["python3", "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

def start_labelstudio():
    """Start Label Studio"""
    print("🏷️ Starting Label Studio...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker not found. Please install Docker first.")
            return
        
        # Start Label Studio with Docker
        subprocess.run([
            "docker", "run", "-it", "--rm",
            "-p", "8080:8080",
            "-v", f"{os.path.abspath('processed')}:/data",
            "-v", f"{os.path.abspath('labelstudio_tasks')}:/labelstudio_tasks",
            "heartexlabs/label-studio:latest",
            "label-studio", "start", "--host", "0.0.0.0", "--port", "8080"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Label Studio stopped")
    except Exception as e:
        print(f"❌ Error starting Label Studio: {e}")

def process_images():
    """Process images using the OCR system"""
    print("📸 Processing images...")
    
    try:
        # Check if there are images to process
        image_dir = Path("toProcess/images")
        if not image_dir.exists() or not any(image_dir.glob("*")):
            print("❌ No images found in toProcess/images/")
            print("Please add some NBA 2K screenshots to process")
            return
        
        # Run OCR processing
        subprocess.run([
            "python3", "automate_2k.py",
            "--input", "toProcess/images",
            "--mode", "legacy"
        ], check=True)
        
        print("✅ Image processing completed!")
        
    except Exception as e:
        print(f"❌ Error processing images: {e}")

def generate_labelstudio_tasks():
    """Generate Label Studio tasks from processed results"""
    print("🏷️ Generating Label Studio tasks...")
    
    try:
        subprocess.run([
            "python3", "labelstudio_task_gen.py",
            "--export-yolo"
        ], check=True)
        
        print("✅ Label Studio tasks generated!")
        
    except Exception as e:
        print(f"❌ Error generating tasks: {e}")

def train_yolo():
    """Train YOLO model"""
    print("🤖 Starting YOLO training...")
    
    try:
        # Check if dataset exists
        data_dir = Path("yolo/data")
        if not data_dir.exists() or not any(data_dir.glob("images/*")):
            print("❌ No dataset found in yolo/data/")
            print("Please generate Label Studio tasks first")
            return
        
        # Start training
        subprocess.run([
            "python3", "yolo/train_yolo.py",
            "--data-dir", "yolo/data",
            "--epochs", "50",
            "--model-size", "n"
        ], check=True)
        
        print("✅ YOLO training completed!")
        
    except Exception as e:
        print(f"❌ Error training YOLO model: {e}")

def open_services():
    """Open web services in browser"""
    print("🌐 Opening web services...")
    
    # Wait a moment for services to start
    time.sleep(2)
    
    # Check if dashboard is running before opening
    try:
        response = subprocess.run(["curl", "-s", "--connect-timeout", "3", "http://localhost:8000/api/status"], 
                                capture_output=True, text=True, timeout=5)
        if response.returncode == 0:
            webbrowser.open("http://localhost:8000")  # Dashboard
            print("✅ Opened dashboard at http://localhost:8000")
        else:
            print("⚠️ Dashboard not ready yet - will open when available")
    except:
        print("⚠️ Could not verify dashboard status")
    
    # Check if Label Studio is running before opening
    try:
        response = subprocess.run(["curl", "-s", "--connect-timeout", "3", "http://localhost:8080"], 
                                capture_output=True, text=True, timeout=5)
        if response.returncode == 0:
            webbrowser.open("http://localhost:8080")  # Label Studio
            print("✅ Opened Label Studio at http://localhost:8080")
        else:
            print("⚠️ Label Studio not ready yet - will open when available")
    except:
        print("⚠️ Could not verify Label Studio status")

def show_status():
    """Show system status"""
    print("📊 NBA 2K OCR System Status")
    print("=" * 40)
    
    # Check processed images
    processed_dir = Path("processed/images")
    if processed_dir.exists():
        processed_count = len(list(processed_dir.glob("*")))
        print(f"📸 Processed images: {processed_count}")
    else:
        print("📸 Processed images: 0")
    
    # Check pending images
    pending_dir = Path("toProcess/images")
    if pending_dir.exists():
        pending_count = len(list(pending_dir.glob("*")))
        print(f"⏳ Pending images: {pending_count}")
    else:
        print("⏳ Pending images: 0")
    
    # Check YOLO model
    model_path = Path("yolo/models/2k_stats_detector/weights/best.pt")
    if model_path.exists():
        print("🤖 YOLO model: Available")
    else:
        print("🤖 YOLO model: Not found")
    
    # Check Label Studio tasks
    tasks_dir = Path("labelstudio_tasks")
    if tasks_dir.exists():
        task_count = len(list(tasks_dir.glob("*.json")))
        print(f"🏷️ Label Studio tasks: {task_count}")
    else:
        print("🏷️ Label Studio tasks: 0")

def main():
    parser = argparse.ArgumentParser(description="NBA 2K OCR System Startup")
    parser.add_argument("command", choices=[
        "setup", "dashboard", "labelstudio", "process", "tasks", 
        "train", "open", "status", "all", "stop"
    ], help="Command to run")
    
    args = parser.parse_args()
    
    print("🏀 NBA 2K OCR System")
    print("=" * 30)
    
    if args.command == "setup":
        print("🔧 Setting up system...")
        if check_dependencies():
            create_directories()
            print("✅ Setup completed!")
        else:
            print("❌ Setup failed - missing dependencies")
    
    elif args.command == "dashboard":
        if check_dependencies():
            start_dashboard()
        else:
            print("❌ Cannot start dashboard - missing dependencies")
    
    elif args.command == "labelstudio":
        start_labelstudio()
    
    elif args.command == "process":
        if check_dependencies():
            process_images()
        else:
            print("❌ Cannot process images - missing dependencies")
    
    elif args.command == "tasks":
        if check_dependencies():
            generate_labelstudio_tasks()
        else:
            print("❌ Cannot generate tasks - missing dependencies")
    
    elif args.command == "train":
        if check_dependencies():
            train_yolo()
        else:
            print("❌ Cannot train model - missing dependencies")
    
    elif args.command == "open":
        open_services()
    
    elif args.command == "status":
        show_status()
    
    elif args.command == "all":
        print("🚀 Starting complete system...")
        if not check_dependencies():
            print("❌ Cannot start system - missing dependencies")
            return
        
        create_directories()
        
        # Check if Docker is available for Label Studio
        docker_available = False
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            docker_available = result.returncode == 0
        except:
            pass
        
        # Start dashboard
        print("🚀 Starting NBA 2K OCR Dashboard...")
        dashboard_process = subprocess.Popen([
            "python3", "dashboard/main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for dashboard to start
        print("⏳ Waiting for dashboard to start...")
        time.sleep(3)
        
        # Check if dashboard is running
        try:
            response = subprocess.run(["curl", "-s", "http://localhost:8000/api/status"], 
                                    capture_output=True, text=True, timeout=5)
            if response.returncode == 0:
                print("✅ Dashboard is running at http://localhost:8000")
            else:
                print("⚠️ Dashboard may not be fully started yet")
        except:
            print("⚠️ Could not verify dashboard status")
        
        # Start Label Studio if Docker is available
        labelstudio_process = None
        if docker_available:
            print("🏷️ Starting Label Studio...")
            try:
                labelstudio_process = subprocess.Popen([
                    "docker", "run", "--rm", "-d",  # Added -d for detached mode
                    "-p", "8080:8080",
                    "-v", f"{os.path.abspath('processed')}:/data",
                    "-v", f"{os.path.abspath('labelstudio_tasks')}:/labelstudio_tasks",
                    "heartexlabs/label-studio:latest",
                    "label-studio", "start", "--host", "0.0.0.0", "--port", "8080"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for Label Studio to start
                time.sleep(5)
                print("✅ Label Studio is starting at http://localhost:8080")
            except Exception as e:
                print(f"⚠️ Could not start Label Studio: {e}")
        else:
            print("⚠️ Docker not available - Label Studio will not be started")
            print("   Install Docker to use Label Studio")
        
        # Open browsers
        print("🌐 Opening web services...")
        open_services()
        
        print("\n🎉 System is running!")
        print("=" * 50)
        print("📊 Dashboard: http://localhost:8000")
        if docker_available:
            print("🏷️ Label Studio: http://localhost:8080")
        print("\n📋 Available commands:")
        print("   • python3 start.py status    - Check system status")
        print("   • python3 start.py process   - Process images")
        print("   • python3 start.py tasks     - Generate Label Studio tasks")
        print("   • python3 start.py train     - Train YOLO model")
        print("\n🛑 Press Ctrl+C to stop dashboard...")
        
        try:
            # Wait for dashboard process (main process)
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            dashboard_process.terminate()
            
            # Stop Label Studio if it was started
            if labelstudio_process:
                try:
                    subprocess.run(["docker", "stop", "$(docker ps -q --filter ancestor=heartexlabs/label-studio:latest)"], 
                                 shell=True, capture_output=True)
                    print("✅ Label Studio stopped")
                except:
                    print("⚠️ Could not stop Label Studio - you may need to stop it manually")
            
            print("✅ All services stopped")
    
    elif args.command == "stop":
        print("🛑 Stopping all services...")
        
        # Stop dashboard processes
        try:
            subprocess.run(["pkill", "-f", "dashboard/main.py"], capture_output=True)
            print("✅ Dashboard stopped")
        except:
            print("⚠️ Could not stop dashboard")
        
        # Stop Label Studio container
        try:
            subprocess.run(["docker", "stop", "$(docker ps -q --filter ancestor=heartexlabs/label-studio:latest)"], 
                         shell=True, capture_output=True)
            print("✅ Label Studio stopped")
        except:
            print("⚠️ Could not stop Label Studio")
        
        # Stop any other related processes
        try:
            subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
            print("✅ Uvicorn processes stopped")
        except:
            pass
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 