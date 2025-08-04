import os
import json
import shutil
import yaml
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
import logging

# Import Label Studio client
try:
    from labelstudio_client import LabelStudioClient
    LABELSTUDIO_AVAILABLE = True
except ImportError:
    LABELSTUDIO_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="NBA 2K OCR Dashboard", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
JSON_DIR = PROCESSED_DIR / "json"
UPLOAD_DIR = BASE_DIR / "toProcess" / "images"

# Create directories if they don't exist
for dir_path in [PROCESSED_DIR, IMAGES_DIR, JSON_DIR, UPLOAD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Load configuration
def load_config():
    try:
        with open(BASE_DIR / "config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

config = load_config()

# Initialize Label Studio client if available
labelstudio_client = None
if LABELSTUDIO_AVAILABLE:
    try:
        labelstudio_client = LabelStudioClient()
        logger.info("✅ Label Studio client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Label Studio client: {e}")
        labelstudio_client = None

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NBA 2K OCR Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
            .upload-area:hover { border-color: #667eea; background-color: #f8f9ff; }
            button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #5a6fd8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏀 NBA 2K OCR Dashboard</h1>
                <p>Process and manage your NBA 2K box score screenshots</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>📸 Processed Images</h3>
                    <p id="processed-count">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>⏳ Pending Images</h3>
                    <p id="pending-count">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>🤖 YOLO Model</h3>
                    <p id="model-status">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>🏷️ Label Studio</h3>
                    <p id="labelstudio-status">Loading...</p>
                </div>
            </div>
            
            <div class="upload-area">
                <h3>📤 Upload New Screenshot</h3>
                <p>Drag and drop or click to upload NBA 2K screenshots</p>
                <input type="file" id="file-input" accept=".png,.jpg,.jpeg" style="display: none;">
                <button onclick="document.getElementById('file-input').click()">Choose File</button>
                <div id="upload-status"></div>
            </div>
            
            <div id="results">
                <h3>📊 Recent Results</h3>
                <div id="results-list">Loading...</div>
            </div>
            
            <div id="labelstudio-section" style="margin-top: 30px;">
                <h3>🏷️ Label Studio Integration</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                    <button onclick="syncLabelStudio()" class="btn btn-primary">Sync with Label Studio</button>
                    <button onclick="openLabelStudio()" class="btn btn-secondary">Open Label Studio</button>
                    <button onclick="getProjects()" class="btn btn-info">Get Projects</button>
                </div>
                <div id="labelstudio-results" style="background: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <p>Label Studio integration ready</p>
                </div>
            </div>
        </div>
        
        <script>
            // Load status on page load
            loadStatus();
            
            // File upload handling
            document.getElementById('file-input').addEventListener('change', function(e) {
                if (e.target.files.length > 0) {
                    uploadFile(e.target.files[0]);
                }
            });
            
            async function loadStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    document.getElementById('processed-count').textContent = data.processed_images;
                    document.getElementById('pending-count').textContent = data.pending_uploads;
                    document.getElementById('model-status').textContent = data.ocr_mode;
                    
                    // Update Label Studio status
                    if (data.labelstudio_available) {
                        document.getElementById('labelstudio-status').textContent = data.labelstudio_status;
                        if (data.labelstudio_status === 'connected') {
                            document.getElementById('labelstudio-status').style.color = 'green';
                        } else if (data.labelstudio_status === 'disconnected') {
                            document.getElementById('labelstudio-status').style.color = 'orange';
                        } else {
                            document.getElementById('labelstudio-status').style.color = 'red';
                        }
                    } else {
                        document.getElementById('labelstudio-status').textContent = 'Not available';
                        document.getElementById('labelstudio-status').style.color = 'gray';
                    }
                    
                    loadResults();
                } catch (error) {
                    console.error('Error loading status:', error);
                }
            }
            
            async function loadResults() {
                try {
                    const response = await fetch('/api/results');
                    const data = await response.json();
                    
                    const resultsList = document.getElementById('results-list');
                    if (data.results && data.results.length > 0) {
                        resultsList.innerHTML = data.results.map(result => 
                            `<div style="background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <strong>${result.image_name}</strong> - ${result.player_count} players (${result.mode})
                            </div>`
                        ).join('');
                    } else {
                        resultsList.innerHTML = '<p>No processed images yet. Upload some screenshots to get started!</p>';
                    }
                } catch (error) {
                    document.getElementById('results-list').innerHTML = '<p>No results available</p>';
                }
            }
            
            async function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);
                
                const statusDiv = document.getElementById('upload-status');
                statusDiv.innerHTML = '<p>Uploading...</p>';
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    statusDiv.innerHTML = `<p style="color: green;">${result.message}</p>`;
                    
                    // Reload status
                    loadStatus();
                } catch (error) {
                    statusDiv.innerHTML = '<p style="color: red;">Upload failed</p>';
                }
            }
            
            // Label Studio functions
            async function syncLabelStudio() {
                try {
                    const response = await fetch('/api/labelstudio/sync', { method: 'POST' });
                    const result = await response.json();
                    
                    const resultsDiv = document.getElementById('labelstudio-results');
                    if (result.success) {
                        resultsDiv.innerHTML = '<p style="color: green;">✅ ' + result.message + '</p>';
                    } else {
                        resultsDiv.innerHTML = '<p style="color: red;">❌ ' + result.message + '</p>';
                    }
                } catch (error) {
                    document.getElementById('labelstudio-results').innerHTML = '<p style="color: red;">❌ Sync failed: ' + error.message + '</p>';
                }
            }
            
            async function openLabelStudio() {
                try {
                    window.open('http://localhost:8080', '_blank');
                    document.getElementById('labelstudio-results').innerHTML = '<p style="color: blue;">🌐 Opening Label Studio...</p>';
                } catch (error) {
                    document.getElementById('labelstudio-results').innerHTML = '<p style="color: red;">❌ Could not open Label Studio</p>';
                }
            }
            
            async function getProjects() {
                try {
                    const response = await fetch('/api/labelstudio/projects');
                    const data = await response.json();
                    
                    const resultsDiv = document.getElementById('labelstudio-results');
                    if (data.projects && data.projects.length > 0) {
                        let html = '<h4>Label Studio Projects:</h4>';
                        data.projects.forEach(project => {
                            html += `<div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                <strong>${project.title}</strong> (ID: ${project.id})<br>
                                <small>Tasks: ${project.task_number || 0}</small>
                            </div>`;
                        });
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = '<p>No projects found in Label Studio</p>';
                    }
                } catch (error) {
                    document.getElementById('labelstudio-results').innerHTML = '<p style="color: red;">❌ Could not get projects: ' + error.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/results")
async def get_all_results():
    """Get all OCR results"""
    results = []
    
    for json_file in JSON_DIR.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                image_name = json_file.stem.replace('_results', '')
                
                results.append({
                    'image_name': image_name,
                    'player_count': len(data.get('players', [])),
                    'mode': data.get('mode', 'legacy'),
                    'timestamp': datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {"results": results}

@app.get("/api/results/{filename}")
async def get_results(filename: str):
    """Get OCR results for a specific file"""
    json_path = JSON_DIR / filename
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {e}")

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Serve processed images"""
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a new image for processing"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # For now, just return success - OCR processing will be handled separately
        return {
            "success": True,
            "message": f"Image {file.filename} uploaded successfully. Use CLI to process: python3 automate_2k.py --input {file_path}"
        }
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {e}")

@app.get("/api/status")
async def get_status():
    """Get system status"""
    status = {
        "processed_images": len(list(JSON_DIR.glob("*.json"))),
        "available_images": len(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))),
        "pending_uploads": len(list(UPLOAD_DIR.glob("*.jpg")) + list(UPLOAD_DIR.glob("*.png"))),
        "ocr_mode": "legacy",
        "labelstudio_available": LABELSTUDIO_AVAILABLE and labelstudio_client is not None
    }
    
    # Add Label Studio status if available
    if labelstudio_client:
        try:
            if labelstudio_client.test_connection():
                status["labelstudio_status"] = "connected"
                projects = labelstudio_client.get_projects()
                status["labelstudio_projects"] = len(projects)
            else:
                status["labelstudio_status"] = "disconnected"
        except Exception as e:
            status["labelstudio_status"] = "error"
            status["labelstudio_error"] = str(e)
    else:
        status["labelstudio_status"] = "unavailable"
    
    return status

@app.get("/api/labelstudio/projects")
async def get_labelstudio_projects():
    """Get Label Studio projects"""
    if not labelstudio_client:
        raise HTTPException(status_code=503, detail="Label Studio client not available")
    
    try:
        projects = labelstudio_client.get_projects()
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting projects: {e}")

@app.post("/api/labelstudio/sync")
async def sync_labelstudio():
    """Sync processed results with Label Studio"""
    if not labelstudio_client:
        raise HTTPException(status_code=503, detail="Label Studio client not available")
    
    try:
        success = labelstudio_client.sync_with_processed_results()
        return {"success": success, "message": "Sync completed" if success else "Sync failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing: {e}")

@app.get("/api/labelstudio/project/{project_id}/stats")
async def get_project_stats(project_id: int):
    """Get Label Studio project statistics"""
    if not labelstudio_client:
        raise HTTPException(status_code=503, detail="Label Studio client not available")
    
    try:
        stats = labelstudio_client.get_project_stats(project_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting project stats: {e}")

@app.post("/api/labelstudio/project/{project_id}/export")
async def export_project(project_id: int, export_type: str = "YOLO"):
    """Export Label Studio project"""
    if not labelstudio_client:
        raise HTTPException(status_code=503, detail="Label Studio client not available")
    
    try:
        if export_type == "YOLO":
            annotations = labelstudio_client.export_annotations(project_id, "YOLO")
            return {"export_type": "YOLO", "annotations": annotations}
        else:
            tasks = labelstudio_client.export_tasks(project_id, export_type)
            return {"export_type": export_type, "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting project: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 