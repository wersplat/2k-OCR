#!/usr/bin/env python3
"""
YOLO Training Script for NBA 2K Stat Region Detection
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOTrainer:
    def __init__(self, data_dir="./data", model_size="n", epochs=100, batch_size=16):
        self.data_dir = Path(data_dir)
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Model paths
        self.weights_dir = Path("./models/2k_stats_detector/weights")
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset paths
        self.dataset_yaml = self.data_dir / "dataset.yaml"
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Class names will be loaded from dataset.yaml
        self.class_names = []
    
    def validate_dataset(self):
        """Validate that the dataset is properly structured"""
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")
        
        # Check for train/val structure
        train_images_dir = self.images_dir / "train"
        val_images_dir = self.images_dir / "val"
        train_labels_dir = self.labels_dir / "train"
        val_labels_dir = self.labels_dir / "val"
        
        if not train_images_dir.exists():
            raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")
        
        if not val_images_dir.exists():
            raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
        
        if not train_labels_dir.exists():
            raise FileNotFoundError(f"Train labels directory not found: {train_labels_dir}")
        
        if not val_labels_dir.exists():
            raise FileNotFoundError(f"Validation labels directory not found: {val_labels_dir}")
        
        # Count files
        train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
        train_labels = list(train_labels_dir.glob("*.txt"))
        val_labels = list(val_labels_dir.glob("*.txt"))
        
        total_images = len(train_images) + len(val_images)
        total_labels = len(train_labels) + len(val_labels)
        
        logging.info(f"Found {len(train_images)} train images and {len(val_images)} validation images")
        logging.info(f"Found {len(train_labels)} train labels and {len(val_labels)} validation labels")
        
        if total_images == 0:
            raise ValueError("No images found in dataset")
        
        if total_labels == 0:
            raise ValueError("No label files found in dataset")
        
        return total_images, total_labels
    
    def load_classes_from_yaml(self):
        """Load class names from dataset.yaml"""
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")
        
        with open(self.dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        if 'names' in dataset_config:
            self.class_names = dataset_config['names']
            logging.info(f"Loaded {len(self.class_names)} classes from dataset.yaml")
        else:
            raise ValueError("No 'names' field found in dataset.yaml")
        
        return self.class_names
    
    def create_dataset_yaml(self):
        """Create or update dataset.yaml file"""
        dataset_config = {
            "path": str(self.data_dir.absolute()),
            "train": "images",
            "val": "images",  # Using same images for validation for now
            "nc": len(self.class_names),
            "names": self.class_names
        }
        
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logging.info(f"Created dataset config: {self.dataset_yaml}")
        return self.dataset_yaml
    
    def split_dataset(self, train_ratio=0.8):
        """Split dataset into train and validation sets"""
        # Create train/val directories
        train_images = self.data_dir / "train" / "images"
        train_labels = self.data_dir / "train" / "labels"
        val_images = self.data_dir / "val" / "images"
        val_labels = self.data_dir / "val" / "labels"
        
        for dir_path in [train_images, train_labels, val_images, val_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        
        # Shuffle and split
        import random
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy files to train/val directories
        for img_file in train_files:
            # Copy image
            shutil.copy2(img_file, train_images / img_file.name)
            # Copy label
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, train_labels / label_file.name)
        
        for img_file in val_files:
            # Copy image
            shutil.copy2(img_file, val_images / img_file.name)
            # Copy label
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, val_labels / label_file.name)
        
        # Update dataset.yaml
        dataset_config = {
            "path": str(self.data_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(self.class_names),
            "names": self.class_names
        }
        
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logging.info(f"Split dataset: {len(train_files)} train, {len(val_files)} validation")
        return len(train_files), len(val_files)
    
    def train_model(self, pretrained_weights=None):
        """Train YOLO model"""
        # Load model
        if pretrained_weights and os.path.exists(pretrained_weights):
            model = YOLO(pretrained_weights)
            logging.info(f"Loaded pretrained weights: {pretrained_weights}")
        else:
            model = YOLO(f'yolov8{self.model_size}.pt')
            logging.info(f"Using base model: yolov8{self.model_size}.pt")
        
        # Training arguments
        train_args = {
            'data': str(self.dataset_yaml),
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': 640,
            'device': 'cpu',  # Use CPU since CUDA is not available
            'project': str(self.weights_dir.parent),
            'name': 'weights',
            'save_period': 10,  # Save every 10 epochs
            'patience': 20,  # Early stopping patience
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'conf': 0.25,
            'iou': 0.45,
            'max_det': 300,
            'half': True,
            'dnn': False,
            'plots': True
        }
        
        logging.info("Starting YOLO training...")
        logging.info(f"Training for {self.epochs} epochs with batch size {self.batch_size}")
        
        # Start training
        results = model.train(**train_args)
        
        # Save best model
        best_model_path = self.weights_dir / "best.pt"
        if os.path.exists(best_model_path):
            logging.info(f"Best model saved to: {best_model_path}")
        
        return results
    
    def validate_model(self, model_path=None):
        """Validate trained model"""
        if model_path is None:
            model_path = self.weights_dir / "best.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data=str(self.dataset_yaml))
        
        logging.info("Validation Results:")
        logging.info(f"mAP50: {results.box.map50:.4f}")
        logging.info(f"mAP50-95: {results.box.map:.4f}")
        logging.info(f"Precision: {results.box.mp:.4f}")
        logging.info(f"Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(self, model_path=None, format='onnx'):
        """Export model to different formats"""
        if model_path is None:
            model_path = self.weights_dir / "best.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Export
        export_path = model.export(format=format)
        logging.info(f"Model exported to: {export_path}")
        
        return export_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for NBA 2K stat detection')
    parser.add_argument('--data-dir', default='./data', help='Dataset directory')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'], default='n', 
                       help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--pretrained', help='Path to pretrained weights')
    parser.add_argument('--split-dataset', action='store_true', help='Split dataset into train/val')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing model')
    parser.add_argument('--export', choices=['onnx', 'torchscript', 'tflite'], 
                       help='Export model to format')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(
        data_dir=args.data_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    try:
        # Validate dataset
        trainer.validate_dataset()
        
        # Load classes from dataset.yaml
        trainer.load_classes_from_yaml()
        
        if args.split_dataset:
            trainer.split_dataset()
        
        if args.validate_only:
            # Only validate existing model
            trainer.validate_model()
        else:
            # Train model
            results = trainer.train_model(args.pretrained)
            
            # Validate trained model
            trainer.validate_model()
        
        # Export if requested
        if args.export:
            trainer.export_model(format=args.export)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 