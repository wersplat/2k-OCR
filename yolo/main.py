#!/usr/bin/env python3
"""
Main script for NBA 2K Box Score Stats Detection using YOLOv8

This script serves as the entry point for using the YOLOv8 integration with the existing OCR pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.yolo_ocr_processor import YOLOOCRProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='NBA 2K Box Score Stats Detection using YOLOv8')
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train YOLOv8 model')
    train_parser.add_argument('--data', type=str, default='yolo/data/dataset.yaml',
                        help='Path to dataset configuration file')
    train_parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    train_parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    train_parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    train_parser.add_argument('--model-size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n, s, m, l, x)')
    train_parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    train_parser.add_argument('--device', type=str, default='',
                        help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    train_parser.add_argument('--project', type=str, default='yolo/models',
                        help='Project directory for saving results')
    train_parser.add_argument('--name', type=str, default='2k_stats_detector',
                        help='Experiment name')
    
    # Detection mode
    detect_parser = subparsers.add_parser('detect', help='Detect stats in images')
    detect_parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv8 model')
    detect_parser.add_argument('--source', type=str, default='./toProcess/images/',
                        help='Path to image or directory of images')
    detect_parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    detect_parser.add_argument('--output', type=str, default='./toProcess/json/',
                        help='Output directory for JSON results')
    detect_parser.add_argument('--visualize', action='store_true',
                        help='Visualize detections')
    detect_parser.add_argument('--vis-output', type=str, default='./visualizations',
                        help='Output directory for visualizations')
    detect_parser.add_argument('--no-fallback', action='store_true',
                        help='Disable fallback to fixed regions')
    
    # Export mode
    export_parser = subparsers.add_parser('export', help='Export trained model')
    export_parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv8 model')
    export_parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'openvino', 'coreml'],
                        help='Export format')
    export_parser.add_argument('--output', type=str, default='yolo/models/exported',
                        help='Output directory for exported model')
    
    return parser.parse_args()

def train_model(args):
    """Train YOLOv8 model"""
    from ultralytics import YOLO
    
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

def detect_stats(args):
    """Detect stats in images using YOLOv8"""
    processor = YOLOOCRProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        use_fallback=not args.no_fallback,
        visualize=args.visualize
    )
    
    if os.path.isdir(args.source):
        processor.process_images(args.source, args.output, args.vis_output if args.visualize else None)
    else:
        visualize_output = os.path.join(args.vis_output, f'{Path(args.source).name}_vis.jpg') if args.visualize else None
        processor.process_image(args.source, args.output, visualize_output)

def export_model(args):
    """Export trained model to different formats"""
    from ultralytics import YOLO
    
    model = YOLO(args.model)
    logging.info(f"Loaded model from {args.model}")
    
    os.makedirs(args.output, exist_ok=True)
    
    # Export the model
    logging.info(f"Exporting model to {args.format} format")
    model.export(format=args.format, output=args.output)
    
    logging.info(f"Model exported to {args.output}")

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'detect':
        detect_stats(args)
    elif args.mode == 'export':
        export_model(args)
    else:
        logging.error("No mode specified. Use 'train', 'detect', or 'export'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
