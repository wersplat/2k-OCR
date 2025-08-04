#!/usr/bin/env python3
"""
Label Studio API Client
Handles authentication and API interactions with Label Studio
"""

import requests
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class LabelStudioClient:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Label Studio client with configuration"""
        self.config = self.load_config(config_path)
        self.base_url = self.config['labelstudio']['auth']['api_url']
        self.token = self.config['labelstudio']['auth']['token']
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Test connection
        self.test_connection()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _get_access_token(self) -> str:
        """Generate access token from personal access token"""
        try:
            response = requests.post(
                f"{self.base_url.replace('/api', '')}/api/token/refresh",
                headers={"Content-Type": "application/json"},
                json={"refresh": self.token},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("access")
            else:
                logger.error(f"Failed to generate access token: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error generating access token: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to Label Studio API"""
        try:
            # First, try to generate an access token from personal access token
            access_token = self._get_access_token()
            
            if access_token:
                # Use the generated access token with Bearer auth
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                
                response = requests.get(f"{self.base_url}/projects/", headers=headers)
                if response.status_code == 200:
                    logger.info("✅ Label Studio API connection successful (Bearer auth with generated token)")
                    return True
            
            # Fallback: try direct personal access token (for SDK compatibility)
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(f"{self.base_url}/projects/", headers=headers)
            if response.status_code == 200:
                logger.info("✅ Label Studio API connection successful (direct personal access token)")
                return True
                
            logger.warning(f"⚠️ Label Studio API connection failed: {response.status_code}")
            logger.warning(f"Response: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Label Studio API connection error: {e}")
            return False
    
    def get_projects(self) -> List[Dict]:
        """Get all projects from Label Studio"""
        try:
            # Get fresh access token
            access_token = self._get_access_token()
            if not access_token:
                logger.error("Failed to get access token")
                return []
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(f"{self.base_url}/projects/", headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return []
    
    def create_project(self, name: str, description: str = "") -> Optional[Dict]:
        """Create a new project in Label Studio"""
        try:
            # Get fresh access token
            access_token = self._get_access_token()
            if not access_token:
                logger.error("Failed to get access token")
                return None
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            project_data = {
                "title": name,
                "description": description,
                "label_config": self.config['labelstudio']['project']['label_config']
            }
            
            response = requests.post(
                f"{self.base_url}/projects/", 
                headers=headers,
                json=project_data
            )
            
            if response.status_code == 201:
                project = response.json()
                logger.info(f"✅ Created project: {project['title']} (ID: {project['id']})")
                return project
            else:
                logger.error(f"Failed to create project: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return None
    
    def get_or_create_project(self, name: str, description: str = "") -> Optional[Dict]:
        """Get existing project or create new one"""
        projects = self.get_projects()
        
        # Check if project already exists
        for project in projects:
            if isinstance(project, dict) and project.get('title') == name:
                logger.info(f"✅ Found existing project: {project.get('title')} (ID: {project.get('id')})")
                return project
        
        # Create new project
        return self.create_project(name, description)
    
    def import_tasks(self, project_id: int, tasks: List[Dict]) -> bool:
        """Import tasks to a project"""
        try:
            # Get fresh access token
            access_token = self._get_access_token()
            if not access_token:
                logger.error("Failed to get access token")
                return False
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/projects/{project_id}/import/",
                headers=headers,
                json=tasks
            )
            
            if response.status_code == 201:
                logger.info(f"✅ Imported {len(tasks)} tasks to project {project_id}")
                return True
            else:
                logger.error(f"Failed to import tasks: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error importing tasks: {e}")
            return False
    
    def export_tasks(self, project_id: int, export_type: str = "JSON") -> List[Dict]:
        """Export tasks from a project"""
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/export/",
                headers=self.headers,
                params={"exportType": export_type}
            )
            
            if response.status_code == 200:
                tasks = response.json()
                logger.info(f"✅ Exported {len(tasks)} tasks from project {project_id}")
                return tasks
            else:
                logger.error(f"Failed to export tasks: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error exporting tasks: {e}")
            return []
    
    def export_annotations(self, project_id: int, export_type: str = "YOLO") -> List[Dict]:
        """Export annotations from a project"""
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/export/",
                headers=self.headers,
                params={"exportType": export_type}
            )
            
            if response.status_code == 200:
                annotations = response.json()
                logger.info(f"✅ Exported {len(annotations)} annotations from project {project_id}")
                return annotations
            else:
                logger.error(f"Failed to export annotations: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            return []
    
    def get_project_stats(self, project_id: int) -> Dict:
        """Get project statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/projects/{project_id}/",
                headers=self.headers
            )
            
            if response.status_code == 200:
                project = response.json()
                stats = {
                    'id': project['id'],
                    'title': project['title'],
                    'task_number': project.get('task_number', 0),
                    'num_tasks_with_annotations': project.get('num_tasks_with_annotations', 0),
                    'useful_annotation_number': project.get('useful_annotation_number', 0),
                    'ground_truth_number': project.get('ground_truth_number', 0),
                    'skipped_annotations_number': project.get('skipped_annotations_number', 0),
                    'total_annotations_number': project.get('total_annotations_number', 0),
                    'total_predictions_number': project.get('total_predictions_number', 0),
                    'comment_authors': project.get('comment_authors', []),
                    'created_at': project.get('created_at'),
                    'updated_at': project.get('updated_at')
                }
                return stats
            else:
                logger.error(f"Failed to get project stats: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            return {}
    
    def delete_project(self, project_id: int) -> bool:
        """Delete a project"""
        try:
            response = requests.delete(
                f"{self.base_url}/projects/{project_id}/",
                headers=self.headers
            )
            
            if response.status_code == 204:
                logger.info(f"✅ Deleted project {project_id}")
                return True
            else:
                logger.error(f"Failed to delete project: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return False
    
    def sync_with_processed_results(self, project_name: str = None) -> bool:
        """Sync processed OCR results with Label Studio project"""
        if project_name is None:
            project_name = self.config['labelstudio']['project']['name']
        
        # Get or create project
        project = self.get_or_create_project(project_name)
        if not project:
            return False
        
        # Load processed results
        processed_dir = Path(self.config['paths']['processed_json'])
        tasks = []
        
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    ocr_results = json.load(f)
                
                # Find corresponding image
                image_name = json_file.stem.replace('_results', '')
                image_path = Path(self.config['paths']['processed_images']) / f"{image_name}.jpg"
                if not image_path.exists():
                    image_path = Path(self.config['paths']['processed_images']) / f"{image_name}.png"
                
                if image_path.exists():
                    # Create task
                    task = {
                        "data": {
                            "image": f"/data/local-files/?d={str(image_path.absolute())}",
                            "filename": image_path.name
                        },
                        "meta": {
                            "ocr_results": ocr_results,
                            "processed_at": datetime.now().isoformat()
                        }
                    }
                    tasks.append(task)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        if tasks:
            if isinstance(project, dict) and 'id' in project:
                return self.import_tasks(project['id'], tasks)
            else:
                logger.error("Invalid project object")
                return False
        else:
            logger.warning("No tasks to import")
            return False

def main():
    """Test the Label Studio client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label Studio API Client")
    parser.add_argument("command", choices=[
        "test", "projects", "create", "sync", "export", "stats"
    ], help="Command to run")
    parser.add_argument("--project-name", default="NBA 2K OCR Project", help="Project name")
    parser.add_argument("--project-id", type=int, help="Project ID")
    
    args = parser.parse_args()
    
    client = LabelStudioClient()
    
    if args.command == "test":
        print("Testing Label Studio connection...")
        if client.test_connection():
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")
    
    elif args.command == "projects":
        print("Getting projects...")
        projects = client.get_projects()
        for project in projects:
            print(f"• {project['title']} (ID: {project['id']})")
    
    elif args.command == "create":
        print(f"Creating project: {args.project_name}")
        project = client.create_project(args.project_name)
        if project:
            print(f"✅ Created project: {project['title']} (ID: {project['id']})")
        else:
            print("❌ Failed to create project")
    
    elif args.command == "sync":
        print("Syncing with processed results...")
        if client.sync_with_processed_results(args.project_name):
            print("✅ Sync completed!")
        else:
            print("❌ Sync failed!")
    
    elif args.command == "export":
        if not args.project_id:
            print("❌ Please provide --project-id")
            return
        print(f"Exporting tasks from project {args.project_id}...")
        tasks = client.export_tasks(args.project_id)
        print(f"✅ Exported {len(tasks)} tasks")
    
    elif args.command == "stats":
        if not args.project_id:
            print("❌ Please provide --project-id")
            return
        print(f"Getting stats for project {args.project_id}...")
        stats = client.get_project_stats(args.project_id)
        if stats:
            print(f"Project: {stats['title']}")
            print(f"Tasks: {stats['task_number']}")
            print(f"Annotated: {stats['num_tasks_with_annotations']}")
            print(f"Total annotations: {stats['total_annotations_number']}")

if __name__ == "__main__":
    main() 