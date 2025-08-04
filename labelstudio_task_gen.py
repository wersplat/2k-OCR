import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LabelStudioTaskGenerator:
    def __init__(self, processed_dir="./processed", output_dir="./labelstudio_tasks"):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Stat region definitions for bounding boxes
        self.stat_regions = {
            'name': {'width': 485, 'height': 81},
            'grade': {'width': 105, 'height': 77},
            'points': {'width': 135, 'height': 77},
            'rebounds': {'width': 135, 'height': 77},
            'assists': {'width': 135, 'height': 77},
            'steals': {'width': 135, 'height': 77},
            'blocks': {'width': 135, 'height': 77},
            'fouls': {'width': 135, 'height': 77},
            'tos': {'width': 135, 'height': 77},
            'FGMFGA': {'width': 205, 'height': 77},
            '3PM3PA': {'width': 205, 'height': 77},
            'FTMFTA': {'width': 205, 'height': 77},
        }
        
        # Player coordinates (normalized for 3840x2160)
        self.player_y_coordinates = [520, 602, 683, 765, 843, 1148, 1233, 1318, 1398, 1479]
        self.base_x_player = 1219
        
        # Player X offsets for each stat
        self.player_x_offsets = {
            'name': 0,
            'grade': 480,
            'points': 620,
            'rebounds': 777,
            'assists': 923,
            'steals': 1075,
            'blocks': 1224,
            'fouls': 1368,
            'tos': 1513,
            'FGMFGA': 1665,
            '3PM3PA': 1898,
            'FTMFTA': 2099,
        }
        
        # Team quarter regions
        self.base_x_team = 317
        self.base_y_team1 = 778
        self.base_y_team2 = 1115
        self.x_offset_team = 110
        self.team_quarter_width = 85
        self.team_quarter_height = 145

    def generate_bounding_boxes(self, image_width=3840, image_height=2160):
        """Generate bounding boxes for all stat regions"""
        boxes = []
        
        # Generate player stat boxes
        for player_num in range(1, 11):  # 10 players
            player_y = self.player_y_coordinates[player_num - 1]
            
            for stat_name, dimensions in self.stat_regions.items():
                x_offset = self.player_x_offsets[stat_name]
                x = self.base_x_player + x_offset
                y = player_y
                width = dimensions['width']
                height = dimensions['height']
                
                # Convert to percentage coordinates for Label Studio
                x_pct = (x / image_width) * 100
                y_pct = (y / image_height) * 100
                width_pct = (width / image_width) * 100
                height_pct = (height / image_height) * 100
                
                boxes.append({
                    'region_name': f'player{player_num}_{stat_name}',
                    'x': x_pct,
                    'y': y_pct,
                    'width': width_pct,
                    'height': height_pct,
                    'label': stat_name,
                    'player_num': player_num
                })
        
        # Generate team quarter boxes
        for quarter in range(1, 5):  # 4 quarters
            x = self.base_x_team + (quarter - 1) * self.x_offset_team
            
            # Team 1 quarters
            y1 = self.base_y_team1
            x_pct1 = (x / image_width) * 100
            y_pct1 = (y1 / image_height) * 100
            width_pct = (self.team_quarter_width / image_width) * 100
            height_pct = (self.team_quarter_height / image_height) * 100
            
            boxes.append({
                'region_name': f'team1_q{quarter}',
                'x': x_pct1,
                'y': y_pct1,
                'width': width_pct,
                'height': height_pct,
                'label': 'team_quarter',
                'team': 1,
                'quarter': quarter
            })
            
            # Team 2 quarters
            y2 = self.base_y_team2
            y_pct2 = (y2 / image_height) * 100
            
            boxes.append({
                'region_name': f'team2_q{quarter}',
                'x': x_pct1,
                'y': y_pct2,
                'width': width_pct,
                'height': height_pct,
                'label': 'team_quarter',
                'team': 2,
                'quarter': quarter
            })
        
        return boxes

    def create_labelstudio_task(self, image_path, ocr_results, bounding_boxes):
        """Create a Label Studio task from OCR results and bounding boxes"""
        task = {
            "data": {
                "image": f"/data/local-files/?d={image_path}",
                "filename": os.path.basename(image_path)
            },
            "annotations": [
                {
                    "result": []
                }
            ],
            "predictions": [
                {
                    "result": []
                }
            ]
        }
        
        # Add bounding boxes with OCR predictions
        for box in bounding_boxes:
            region_name = box['region_name']
            
            # Find corresponding OCR result
            ocr_value = self.get_ocr_value(ocr_results, region_name)
            
            # Create Label Studio result item
            result_item = {
                "id": f"result_{len(task['predictions'][0]['result'])}",
                "type": "rectanglelabels",
                "value": {
                    "x": box['x'],
                    "y": box['y'],
                    "width": box['width'],
                    "height": box['height'],
                    "rectanglelabels": [box['label']]
                },
                "from_name": "label",
                "to_name": "image",
                "meta": {
                    "region_name": region_name,
                    "ocr_value": ocr_value
                }
            }
            
            task['predictions'][0]['result'].append(result_item)
        
        return task

    def get_ocr_value(self, ocr_results, region_name):
        """Extract OCR value for a specific region"""
        if region_name.startswith('player'):
            parts = region_name.split('_')
            player_num = int(parts[0][6:])
            stat_name = parts[1]
            
            # Find player in results
            for player in ocr_results.get('players', []):
                if player.get('player_number') == player_num:
                    if stat_name == 'FGMFGA':
                        return f"{player.get('FGM', '0')}/{player.get('FGA', '0')}"
                    elif stat_name == '3PM3PA':
                        return f"{player.get('3PM', '0')}/{player.get('3PA', '0')}"
                    elif stat_name == 'FTMFTA':
                        return f"{player.get('FTM', '0')}/{player.get('FTA', '0')}"
                    else:
                        return str(player.get(stat_name, '0'))
        
        elif region_name.startswith('team'):
            parts = region_name.split('_')
            team_num = parts[0][4:]
            quarter = parts[1][1:]
            
            team_key = f"team{team_num}_quarters"
            quarter_key = f"quarter_{quarter}"
            
            return str(ocr_results.get('teams', {}).get(team_key, {}).get(quarter_key, '0'))
        
        return ''

    def generate_tasks_from_processed(self):
        """Generate Label Studio tasks from all processed OCR results"""
        json_dir = self.processed_dir / "json"
        images_dir = self.processed_dir / "images"
        
        if not json_dir.exists():
            logging.error(f"JSON directory not found: {json_dir}")
            return
        
        tasks = []
        bounding_boxes = self.generate_bounding_boxes()
        
        for json_file in json_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    ocr_results = json.load(f)
                
                # Find corresponding image
                image_name = json_file.stem.replace('_results', '')
                image_path = None
                
                for ext in ['.jpg', '.png', '.jpeg']:
                    potential_path = images_dir / f"{image_name}{ext}"
                    if potential_path.exists():
                        image_path = str(potential_path)
                        break
                
                if not image_path:
                    logging.warning(f"No image found for {image_name}")
                    continue
                
                # Create task
                task = self.create_labelstudio_task(image_path, ocr_results, bounding_boxes)
                tasks.append(task)
                
                logging.info(f"Generated task for {image_name}")
                
            except Exception as e:
                logging.error(f"Error processing {json_file}: {e}")
        
        return tasks

    def save_tasks(self, tasks, filename=None):
        """Save tasks to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"labelstudio_tasks_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        logging.info(f"Saved {len(tasks)} tasks to {output_path}")
        return output_path

    def create_labelstudio_config(self):
        """Create Label Studio configuration file"""
        config = {
            "label_config": """
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="name" background="red"/>
    <Label value="grade" background="blue"/>
    <Label value="points" background="green"/>
    <Label value="rebounds" background="yellow"/>
    <Label value="assists" background="orange"/>
    <Label value="steals" background="purple"/>
    <Label value="blocks" background="pink"/>
    <Label value="fouls" background="brown"/>
    <Label value="tos" background="gray"/>
    <Label value="FGMFGA" background="cyan"/>
    <Label value="3PM3PA" background="magenta"/>
    <Label value="FTMFTA" background="lime"/>
    <Label value="team_quarter" background="navy"/>
  </RectangleLabels>
</View>
            """.strip(),
            "input_path": str(self.processed_dir / "images"),
            "output_path": str(self.output_dir)
        }
        
        config_path = self.output_dir / "labelstudio_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Created Label Studio config: {config_path}")
        return config_path

    def export_for_yolo_training(self, tasks, output_dir="./yolo/data"):
        """Export Label Studio tasks for YOLO training"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO dataset structure
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Class mapping
        class_mapping = {
            'name': 0, 'grade': 1, 'points': 2, 'rebounds': 3, 'assists': 4,
            'steals': 5, 'blocks': 6, 'fouls': 7, 'tos': 8, 'FGMFGA': 9,
            '3PM3PA': 10, 'FTMFTA': 11, 'team_quarter': 12
        }
        
        for i, task in enumerate(tasks):
            image_path = task['data']['image'].replace('/data/local-files/?d=', '')
            image_name = os.path.basename(image_path)
            
            # Copy image to YOLO dataset
            source_image = Path(image_path)
            if source_image.exists():
                dest_image = images_dir / image_name
                shutil.copy2(source_image, dest_image)
                
                # Create YOLO annotation file
                label_file = labels_dir / f"{os.path.splitext(image_name)[0]}.txt"
                
                with open(label_file, 'w') as f:
                    for result in task['predictions'][0]['result']:
                        value = result['value']
                        label = value['rectanglelabels'][0]
                        
                        if label in class_mapping:
                            class_id = class_mapping[label]
                            
                            # Convert to YOLO format (center_x, center_y, width, height)
                            center_x = (value['x'] + value['width'] / 2) / 100
                            center_y = (value['y'] + value['height'] / 2) / 100
                            width = value['width'] / 100
                            height = value['height'] / 100
                            
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                logging.info(f"Exported YOLO annotation for {image_name}")
        
        # Create dataset.yaml
        dataset_config = {
            "path": str(output_path.absolute()),
            "train": "images",
            "val": "images",
            "nc": len(class_mapping),
            "names": list(class_mapping.keys())
        }
        
        yaml_path = output_path / "dataset.yaml"
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logging.info(f"Created YOLO dataset config: {yaml_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate Label Studio tasks from OCR results')
    parser.add_argument('--processed-dir', default='./processed', help='Directory with processed OCR results')
    parser.add_argument('--output-dir', default='./labelstudio_tasks', help='Output directory for tasks')
    parser.add_argument('--export-yolo', action='store_true', help='Export for YOLO training')
    parser.add_argument('--yolo-output', default='./yolo/data', help='YOLO dataset output directory')
    
    args = parser.parse_args()
    
    generator = LabelStudioTaskGenerator(args.processed_dir, args.output_dir)
    
    # Generate tasks
    tasks = generator.generate_tasks_from_processed()
    
    if tasks:
        # Save tasks
        output_path = generator.save_tasks(tasks)
        
        # Create Label Studio config
        config_path = generator.create_labelstudio_config()
        
        # Export for YOLO if requested
        if args.export_yolo:
            yolo_path = generator.export_for_yolo_training(tasks, args.yolo_output)
            logging.info(f"YOLO dataset exported to: {yolo_path}")
        
        logging.info(f"Generated {len(tasks)} Label Studio tasks")
        logging.info(f"Tasks saved to: {output_path}")
        logging.info(f"Config saved to: {config_path}")
    else:
        logging.warning("No tasks generated. Check if processed OCR results exist.")

if __name__ == "__main__":
    main()