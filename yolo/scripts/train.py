#!/usr/bin/env python3
"""
YOLOv8 Training Script for NBA 2K Box Score Stats Detection

This script trains a YOLOv8 model to detect stat blocks in NBA 2K box score screenshots.
"""

import os
import argparse
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for NBA 2K box score stats detection')
    parser.add_argument('--data', type=str, default='../data/dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--project', type=str, default='../models',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='2k_stats_detector',
                        help='Experiment name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create project directory if it doesn't exist
    os.makedirs(args.project, exist_ok=True)
    
    # Initialize model
    if args.pretrained:
        model = YOLO(f'yolov8{args.model_size}.pt')
        logging.info(f"Loaded pretrained YOLOv8{args.model_size} model")
    else:
        model = YOLO(f'yolov8{args.model_size}.yaml')
        logging.info(f"Created new YOLOv8{args.model_size} model")
    
    # Train the model
    logging.info(f"Starting training for {args.epochs} epochs with batch size {args.batch}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        project=args.project,
        name=args.name,
        device=args.device
    )
    
    # Validate the model
    logging.info("Validating model")
    metrics = model.val()
    logging.info(f"Validation metrics: {metrics}")
    
    logging.info(f"Training complete. Model saved to {os.path.join(args.project, args.name)}")

if __name__ == "__main__":
    main()
