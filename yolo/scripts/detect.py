#!/usr/bin/env python3
"""
YOLOv8 Inference Script for NBA 2K Box Score Stats Detection

This script uses a trained YOLOv8 model to detect stat blocks in NBA 2K box score screenshots.
"""

import os
import argparse
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Detect stat blocks in NBA 2K box score screenshots')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv8 model')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--output', type=str, default='../output',
                        help='Output directory')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save cropped detections')
    parser.add_argument('--save-json', action='store_true',
                        help='Save detection results as JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize detections')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(args.model)
    logging.info(f"Loaded model from {args.model}")
    
    # Get source path
    source = Path(args.source)
    if source.is_dir():
        image_paths = [p for p in source.glob('*.jpg') or source.glob('*.png') or source.glob('*.jpeg')]
    else:
        image_paths = [source]
    
    for img_path in image_paths:
        logging.info(f"Processing {img_path}")
        
        # Run inference
        results = model(str(img_path), conf=args.conf, iou=args.iou)
        
        # Process results
        result = results[0]  # Get first result (only one image)
        
        # Get class names from model
        class_names = model.names
        
        # Prepare output paths
        img_name = img_path.stem
        vis_path = output_dir / f"{img_name}_vis.jpg"
        json_path = output_dir / f"{img_name}_detections.json"
        crops_dir = output_dir / "crops" / img_name
        
        if args.save_crops:
            crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original image for visualization and cropping
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]
        
        # Prepare detections list for JSON output
        detections = []
        
        # Process each detection
        for i, det in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls_id = int(det.cls[0])
            cls_name = class_names[cls_id]
            
            # Add detection to list
            detections.append({
                'id': i,
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'width': x2 - x1,
                'height': y2 - y1,
                'center_x': (x1 + x2) // 2,
                'center_y': (y1 + y2) // 2
            })
            
            # Save crop if requested
            if args.save_crops:
                crop = img[y1:y2, x1:x2]
                crop_path = crops_dir / f"{cls_name}_{i}.jpg"
                cv2.imwrite(str(crop_path), crop)
            
            # Draw bounding box for visualization
            if args.visualize:
                # Define colors for different classes
                colors = {
                    'player_name': (0, 255, 0),    # Green
                    'grade': (255, 0, 0),          # Blue
                    'points': (0, 0, 255),         # Red
                    'rebounds': (255, 255, 0),     # Cyan
                    'assists': (255, 0, 255),      # Magenta
                    'steals': (0, 255, 255),       # Yellow
                    'blocks': (128, 0, 0),         # Dark Blue
                    'fouls': (0, 128, 0),          # Dark Green
                    'turnovers': (0, 0, 128),      # Dark Red
                    'fgm_fga': (128, 128, 0),      # Dark Cyan
                    '3pm_3pa': (128, 0, 128),      # Dark Magenta
                    'ftm_fta': (0, 128, 128)       # Dark Yellow
                }
                color = colors.get(cls_name, (255, 255, 255))  # Default to white
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization if requested
        if args.visualize:
            cv2.imwrite(str(vis_path), img)
            logging.info(f"Saved visualization to {vis_path}")
        
        # Save JSON if requested
        if args.save_json:
            with open(json_path, 'w') as f:
                json.dump({
                    'image_path': str(img_path),
                    'image_width': img_width,
                    'image_height': img_height,
                    'detections': detections
                }, f, indent=2)
            logging.info(f"Saved detections to {json_path}")
    
    logging.info("Processing complete")

if __name__ == "__main__":
    main()
