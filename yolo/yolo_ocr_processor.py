#!/usr/bin/env python3
"""
YOLOv8 OCR Processor for NBA 2K Box Score Stats

This module integrates YOLOv8 object detection with EasyOCR for processing NBA 2K box score screenshots.
It replaces the fixed cropping logic with dynamic detection of stat blocks.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
import easyocr
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants from original OCR processor
PLAYERS = 10
JSON_OUTPUT_FOLDER = './toProcess/json/'
IMAGE_OUTPUT_FOLDER = './processed/images/'

# Original fixed region coordinates for fallback
BASE_X_PLAYER = 1219
PLAYER_Y_COORDINATES = [520, 602, 683, 765, 843, 1148, 1233, 1318, 1398, 1479]

PLAYER_WIDTHS = {
    'name': 485,
    'grade': 105,
    'points': 135,
    'rebounds': 135,
    'assists': 135,
    'steals': 135,
    'blocks': 135,
    'fouls': 135,
    'tos': 135,
    'FGMFGA': 205,
    '3PM3PA': 205,
    'FTMFTA': 205,
}

PLAYER_X_OFFSETS = {
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

PLAYER_HEIGHTS = 77
PLAYER_NAME_HEIGHT = 81

# Mapping from YOLO class names to original stat names
YOLO_TO_STAT_MAP = {
    'player_name': 'name',
    'grade': 'grade',
    'points': 'points',
    'rebounds': 'rebounds',
    'assists': 'assists',
    'steals': 'steals',
    'blocks': 'blocks',
    'fouls': 'fouls',
    'turnovers': 'tos',
    'fgm_fga': 'FGMFGA',
    '3pm_3pa': '3PM3PA',
    'ftm_fta': 'FTMFTA'
}

# Visualization colors for different classes
CLASS_COLORS = {
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

class YOLOOCRProcessor:
    def __init__(self, model_path, conf_threshold=0.25, use_fallback=True, visualize=False):
        """
        Initialize the YOLO OCR Processor
        
        Args:
            model_path (str): Path to the trained YOLOv8 model
            conf_threshold (float): Confidence threshold for detections
            use_fallback (bool): Whether to use fallback to fixed regions if detection fails
            visualize (bool): Whether to visualize and save detection results
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.use_fallback = use_fallback
        self.visualize = visualize
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA available: {self.cuda_available}")
        if self.cuda_available:
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Running on CPU")
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            logging.info(f"Loaded YOLOv8 model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLOv8 model: {e}")
            if self.use_fallback:
                logging.warning("Will use fallback to fixed regions")
            else:
                raise
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=self.cuda_available)
        logging.info(f"Initialized EasyOCR reader (GPU: {self.cuda_available})")
        
        # Generate fixed regions for fallback
        self.fixed_regions = self.generate_fixed_regions()
    
    def generate_fixed_regions(self):
        """Generate fixed regions for fallback, based on original OCR processor"""
        regions = []

        # Generate player regions
        for i in range(1, PLAYERS + 1):
            current_y = PLAYER_Y_COORDINATES[i - 1]
            regions.append({
                'name': f'player{i}_name',
                'x': BASE_X_PLAYER,
                'y': current_y,
                'width': PLAYER_WIDTHS['name'],
                'height': PLAYER_NAME_HEIGHT
            })
            for stat, width in PLAYER_WIDTHS.items():
                if stat != 'name':
                    regions.append({
                        'name': f'player{i}_{stat}',
                        'x': BASE_X_PLAYER + PLAYER_X_OFFSETS[stat],
                        'y': current_y,
                        'width': width,
                        'height': PLAYER_HEIGHTS
                    })

        return regions
    
    def detect_regions(self, image_path):
        """
        Detect stat regions in an image using YOLOv8
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            list: List of detected regions with name, coordinates, and confidence
        """
        try:
            # Run inference
            results = self.model(image_path, conf=self.conf_threshold)
            result = results[0]  # Get first result (only one image)
            
            # Get class names from model
            class_names = self.model.names
            
            # Process detections
            detections = []
            for i, det in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf[0])
                cls_id = int(det.cls[0])
                cls_name = class_names[cls_id]
                
                # Extract player number based on Y position
                player_num = self.estimate_player_number(y1, y2)
                
                # Map YOLO class name to original stat name
                stat_name = YOLO_TO_STAT_MAP.get(cls_name, cls_name)
                
                # Create region dict
                region = {
                    'name': f'player{player_num}_{stat_name}',
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'confidence': conf,
                    'class_name': cls_name
                }
                detections.append(region)
            
            return detections
            
        except Exception as e:
            logging.error(f"Error in YOLO detection: {e}")
            return []
    
    def estimate_player_number(self, y1, y2):
        """
        Estimate player number based on Y position
        
        Args:
            y1 (int): Top Y coordinate
            y2 (int): Bottom Y coordinate
            
        Returns:
            int: Estimated player number (1-10)
        """
        center_y = (y1 + y2) // 2
        
        # Calculate distances to each player row
        distances = [abs(center_y - y) for y in PLAYER_Y_COORDINATES]
        
        # Find the closest player row
        closest_idx = distances.index(min(distances))
        return closest_idx + 1
    
    def crop_regions(self, image_path, regions):
        """
        Crop regions from an image
        
        Args:
            image_path (str): Path to the image
            regions (list): List of regions to crop
            
        Returns:
            tuple: (cropped_images, region_names)
        """
        image = Image.open(image_path)
        width, height = image.size
        
        cropped_images = []
        region_names = []
        
        for region in regions:
            left = region['x']
            upper = region['y']
            right = left + region['width']
            lower = upper + region['height']
            
            # Ensure coordinates are within image bounds
            left = max(0, left)
            upper = max(0, upper)
            right = min(width, right)
            lower = min(height, lower)
            
            cropped_image = image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)
            region_names.append(region['name'])
        
        return cropped_images, region_names
    
    def crop_fixed_regions(self, image_path):
        """
        Crop fixed regions from an image (fallback method)
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            tuple: (cropped_images, region_names)
        """
        image = Image.open(image_path)
        width, height = image.size
        
        cropped_images = []
        region_names = []
        
        for region in self.fixed_regions:
            left = int(region['x'] * width / 3840)
            upper = int(region['y'] * height / 2160)
            right = left + int(region['width'] * width / 3840)
            lower = upper + int(region['height'] * height / 2160)
            
            cropped_image = image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)
            region_names.append(region['name'])
        
        return cropped_images, region_names
    
    def visualize_detections(self, image_path, regions, output_path):
        """
        Visualize detections on the image
        
        Args:
            image_path (str): Path to the image
            regions (list): List of detected regions
            output_path (str): Path to save the visualization
        """
        img = cv2.imread(image_path)
        
        for region in regions:
            x = region['x']
            y = region['y']
            w = region['width']
            h = region['height']
            cls_name = region['class_name']
            conf = region.get('confidence', 1.0)
            
            # Get color for this class
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization
        cv2.imwrite(output_path, img)
        logging.info(f"Saved visualization to {output_path}")
    
    def detect_text_in_image(self, image, region_name, allowlist=None):
        """
        Detect text in an image using EasyOCR
        
        Args:
            image (PIL.Image): Image to process
            region_name (str): Name of the region
            allowlist (str): Allowlist of characters
            
        Returns:
            list: List of detected texts
        """
        # Convert PIL image to OpenCV format
        img_cropped = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply scaling and blurring only to numeric regions
        if "name" not in region_name and "grade" not in region_name:
            scale_factor = 2
            upscaled = cv2.resize(img_cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            blur = cv2.blur(upscaled, (5, 5))
            img_cropped = blur
        
        # Perform OCR on the processed image
        result = self.reader.readtext(img_cropped, detail=0, allowlist=allowlist, text_threshold=0.3)
        
        detected_texts = ['0' if text.strip() == '' else text for text in result]
        
        # Remove spaces from numeric stats
        if "name" not in region_name:
            detected_texts = [text.replace(' ', '') for text in detected_texts]
        
        return detected_texts
    
    @staticmethod
    def filter_text(text):
        """Filter text to remove unwanted characters"""
        filtered_text = text.replace('.', '')
        return filtered_text.strip()
    
    @staticmethod
    def fix_slash_in_stats(stat):
        """Fix slash in stats like FGM/FGA"""
        stat = stat.strip()
        
        if len(stat) == 3 and stat[1] == '1':
            return f"{stat[0]}/{stat[2]}"
        if len(stat) == 4 and stat[1] == '1':
            return f"{stat[0]}/{stat[2:]}"
        if len(stat) == 5 and stat[2] == '1':
            return f"{stat[0:2]}/{stat[3:]}"
        
        return stat
    
    @staticmethod
    def get_allowlist(region_name):
        """Get allowlist of characters for OCR based on region name"""
        if "FGMFGA" in region_name or "3PM3PA" in region_name or "FTMFTA" in region_name:
            return '0123456789/'
        if "grade" in region_name:
            return 'ABCDF+-'
        if "player" in region_name and "name" not in region_name:
            return '0123456789'
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-_/ '
    
    @staticmethod
    def correct_common_errors(text):
        """Correct common OCR errors"""
        corrections = {
            "Al Player": "AI Player",
            "Al Player 3": "AI Player",
        }
        return corrections.get(text, text)
    
    @staticmethod
    def get_team(player_number):
        """Get team based on player number"""
        if 1 <= player_number <= 5:
            return 'team1'
        elif 6 <= player_number <= 10:
            return 'team2'
        return 'unknown'
    
    @staticmethod
    def get_position(player_number):
        """Get position based on player number"""
        positions = {
            1: 'PG', 2: 'SG', 3: 'SF', 4: 'PF', 5: 'C',
            6: 'PG', 7: 'SG', 8: 'SF', 9: 'PF', 10: 'C'
        }
        return positions.get(player_number, '')
    
    def format_ocr_results(self, ocr_results, region_names):
        """
        Format OCR results into structured output
        
        Args:
            ocr_results (list): List of OCR results
            region_names (list): List of region names
            
        Returns:
            dict: Formatted output
        """
        formatted_output = {"players": [], "teams": {"team1_quarters": {}, "team2_quarters": {}}}
        current_player = None
        
        for i, region_name in enumerate(region_names):
            text = " ".join(ocr_results[i]).strip()
            
            if region_name.startswith("player"):
                parts = region_name.split('_')
                player_number = int(parts[0][6:])
                stat_name = parts[1]
                
                if stat_name == "name":
                    text = self.correct_common_errors(text)
                    current_player = {
                        "player_number": player_number,
                        "position": self.get_position(player_number),
                        "team": self.get_team(player_number),
                        "name": text
                    }
                    formatted_output["players"].append(current_player)
                
                if current_player is not None:
                    if stat_name == "FGMFGA":
                        text = self.fix_slash_in_stats(text)
                        if '/' in text:
                            fgm, fga = text.split('/')
                            current_player['FGM'] = fgm if fgm else '0'
                            current_player['FGA'] = fga if fga else '0'
                        else:
                            current_player['FGM'] = '0'
                            current_player['FGA'] = '0'
                    elif stat_name == "3PM3PA":
                        text = self.fix_slash_in_stats(text)
                        if '/' in text:
                            three_pm, three_pa = text.split('/')
                            current_player['3PM'] = three_pm if three_pm else '0'
                            current_player['3PA'] = three_pa if three_pa else '0'
                        else:
                            current_player['3PM'] = '0'
                            current_player['3PA'] = '0'
                    elif stat_name == "FTMFTA":
                        text = self.fix_slash_in_stats(text)
                        if '/' in text:
                            ftm, fta = text.split('/')
                            current_player['FTM'] = ftm if ftm else '0'
                            current_player['FTA'] = fta if fta else '0'
                        else:
                            current_player['FTM'] = '0'
                            current_player['FTA'] = '0'
                    elif stat_name == "grade":
                        current_player[stat_name] = text if text else 'C'
                    else:
                        current_player[stat_name] = text if text else '0'
            
            elif "team1_q" in region_name:
                quarter_number = region_name[-1]
                formatted_output["teams"]["team1_quarters"][f"quarter_{quarter_number}"] = text if text else '0'
            
            elif "team2_q" in region_name:
                quarter_number = region_name[-1]
                formatted_output["teams"]["team2_quarters"][f"quarter_{quarter_number}"] = text if text else '0'
        
        return formatted_output
    
    def process_image(self, image_path, output_folder=None, visualize_output=None):
        """
        Process a single image
        
        Args:
            image_path (str): Path to the image
            output_folder (str): Folder to save JSON output
            visualize_output (str): Path to save visualization
            
        Returns:
            dict: Formatted results
        """
        logging.info(f'Processing file: {image_path}')
        
        # Detect regions using YOLO
        detected_regions = self.detect_regions(image_path)
        
        # Check if we have enough detections
        expected_regions = PLAYERS * len(YOLO_TO_STAT_MAP)
        detected_count = len(detected_regions)
        
        if detected_count < expected_regions * 0.5 and self.use_fallback:
            logging.warning(f"Only detected {detected_count}/{expected_regions} regions. Using fallback.")
            cropped_images, region_names = self.crop_fixed_regions(image_path)
        else:
            logging.info(f"Detected {detected_count}/{expected_regions} regions.")
            
            # Sort regions by player number and stat type for consistent processing
            detected_regions.sort(key=lambda r: (int(r['name'].split('_')[0][6:]), r['name'].split('_')[1]))
            
            # Crop detected regions
            cropped_images, region_names = self.crop_regions(image_path, detected_regions)
            
            # Visualize if requested
            if self.visualize and visualize_output:
                self.visualize_detections(image_path, detected_regions, visualize_output)
        
        # Perform OCR on cropped regions
        ocr_results = []
        for cropped_image, region_name in zip(cropped_images, region_names):
            allowlist = self.get_allowlist(region_name)
            ocr_text = self.detect_text_in_image(cropped_image, region_name, allowlist=allowlist)
            ocr_results.append(ocr_text)
        
        # Format results
        formatted_results = self.format_ocr_results(ocr_results, region_names)
        
        # Generate hash of the formatted results
        results_json_str = json.dumps(formatted_results, sort_keys=True)
        results_hash = hashlib.sha256(results_json_str.encode('utf-8')).hexdigest()
        formatted_results['hash'] = results_hash
        
        # Save results if output folder is provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_json_path = os.path.join(output_folder, f'{Path(image_path).name}_results.json')
            with open(output_json_path, 'w') as json_file:
                json.dump(formatted_results, json_file, indent=4)
            logging.info(f'Formatted results saved to {output_json_path}')
        
        return formatted_results
    
    def process_images(self, input_folder, output_folder=None, visualize_folder=None):
        """
        Process multiple images in a folder
        
        Args:
            input_folder (str): Folder containing images
            output_folder (str): Folder to save JSON output
            visualize_folder (str): Folder to save visualizations
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        if visualize_folder and self.visualize:
            os.makedirs(visualize_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(input_folder, filename)
                
                if visualize_folder and self.visualize:
                    visualize_output = os.path.join(visualize_folder, f'{filename}_vis.jpg')
                else:
                    visualize_output = None
                
                self.process_image(input_image_path, output_folder, visualize_output)
                
                # Move processed image to output folder if specified
                if IMAGE_OUTPUT_FOLDER:
                    os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)
                    output_image_path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
                    shutil.copy(input_image_path, output_image_path)
                    logging.info(f'Moved processed image to {output_image_path}')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NBA 2K box score screenshots using YOLOv8 and EasyOCR')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLOv8 model')
    parser.add_argument('--input', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--output', type=str, default=JSON_OUTPUT_FOLDER, help='Output directory for JSON results')
    parser.add_argument('--visualize', action='store_true', help='Visualize detections')
    parser.add_argument('--vis-output', type=str, default='./visualizations', help='Output directory for visualizations')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--no-fallback', action='store_true', help='Disable fallback to fixed regions')
    
    args = parser.parse_args()
    
    processor = YOLOOCRProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        use_fallback=not args.no_fallback,
        visualize=args.visualize
    )
    
    if os.path.isdir(args.input):
        processor.process_images(args.input, args.output, args.vis_output if args.visualize else None)
    else:
        visualize_output = os.path.join(args.vis_output, f'{Path(args.input).name}_vis.jpg') if args.visualize else None
        processor.process_image(args.input, args.output, visualize_output)


if __name__ == "__main__":
    main()
