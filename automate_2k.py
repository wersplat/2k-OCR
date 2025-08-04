import os
import json
import shutil
import re
import numpy as np
from PIL import Image
import easyocr
import torch
import logging
import hashlib
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PLAYERS = 10
TEAM1_QUARTERS = 4
TEAM2_QUARTERS = 4

JSON_OUTPUT_FOLDER = './processed/json/'
IMAGE_INPUT_FOLDER = './toProcess/images/'
IMAGE_OUTPUT_FOLDER = './processed/images/'
YOLO_MODEL_PATH = './yolo/models/2k_stats_detector/weights/best.pt'

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

BASE_X_TEAM = 317
BASE_Y_TEAM1 = 778
BASE_Y_TEAM2 = 1115
X_OFFSET_TEAM = 110
TEAM_QUARTER_WIDTH = 85
TEAM_QUARTER_HEIGHT = 145

# Check GPU availability
cuda_available = torch.cuda.is_available()
logging.info(f"CUDA available: {cuda_available}")
if cuda_available:
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Running on CPU")

class YOLODetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model for stat region detection"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logging.info(f"Loaded YOLO model from {self.model_path}")
            else:
                logging.warning(f"YOLO model not found at {self.model_path}, using default YOLOv8n")
                self.model = YOLO('yolov8n.pt')
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_regions(self, image_path):
        """Detect stat regions using YOLO model"""
        if not self.model:
            logging.warning("YOLO model not available, falling back to legacy mode")
            return None
        
        try:
            results = self.model(image_path)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': cls
                        })
            
            return detections
        except Exception as e:
            logging.error(f"Error in YOLO detection: {e}")
            return None

class OCRProcessor:
    def __init__(self, mode='legacy', yolo_model_path=None):
        self.mode = mode
        self.yolo_detector = YOLODetector(yolo_model_path) if mode == 'yolo' else None
        self.regions = self.generate_regions()

    def generate_regions(self):
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

        # Generate team regions
        for i in range(1, TEAM1_QUARTERS + 1):
            regions.append({
                'name': f'team1_q{i}',
                'x': BASE_X_TEAM + (i - 1) * X_OFFSET_TEAM,
                'y': BASE_Y_TEAM1,
                'width': TEAM_QUARTER_WIDTH,
                'height': TEAM_QUARTER_HEIGHT
            })

        for i in range(1, TEAM2_QUARTERS + 1):
            regions.append({
                'name': f'team2_q{i}',
                'x': BASE_X_TEAM + (i - 1) * X_OFFSET_TEAM,
                'y': BASE_Y_TEAM2,
                'width': TEAM_QUARTER_WIDTH,
                'height': TEAM_QUARTER_HEIGHT
            })

        return regions

    def crop_and_save_regions(self, image_path):
        image = Image.open(image_path)
        width, height = image.size

        if self.mode == 'yolo' and self.yolo_detector:
            # Use YOLO detections if available
            detections = self.yolo_detector.detect_regions(image_path)
            if detections:
                return self.crop_yolo_regions(image, detections)
        
        # Fall back to legacy mode
        return self.crop_legacy_regions(image, width, height)

    def crop_legacy_regions(self, image, width, height):
        """Legacy region cropping based on fixed coordinates"""
        cropped_images = []
        region_names = []

        for region in self.regions:
            left = int(region['x'] * width / 3840)
            upper = int(region['y'] * height / 2160)
            right = left + int(region['width'] * width / 3840)
            lower = upper + int(region['height'] * height / 2160)
            cropped_image = image.crop((left, upper, right, lower))
            cropped_images.append(cropped_image)
            region_names.append(region['name'])

        return cropped_images, region_names

    def crop_yolo_regions(self, image, detections):
        """Crop regions based on YOLO detections"""
        cropped_images = []
        region_names = []
        
        # Sort detections by confidence and class
        sorted_detections = sorted(detections, key=lambda x: (x['class'], -x['confidence']))
        
        for i, detection in enumerate(sorted_detections):
            x1, y1, x2, y2 = detection['bbox']
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_image)
            
            # Map detection class to region name
            region_name = self.map_detection_to_region(detection['class'], i)
            region_names.append(region_name)
        
        return cropped_images, region_names

    def map_detection_to_region(self, class_id, index):
        """Map YOLO detection class to region name"""
        # This mapping should be based on your YOLO training classes
        # For now, using a simple mapping - adjust based on your training
        stat_types = ['name', 'grade', 'points', 'rebounds', 'assists', 'steals', 
                     'blocks', 'fouls', 'tos', 'FGMFGA', '3PM3PA', 'FTMFTA']
        
        player_num = (index // len(stat_types)) + 1
        stat_type = stat_types[index % len(stat_types)]
        
        if player_num <= 10:
            return f'player{player_num}_{stat_type}'
        else:
            team_num = 1 if player_num <= 15 else 2
            quarter = ((player_num - 11) % 4) + 1
            return f'team{team_num}_q{quarter}'

    @staticmethod
    def filter_text(text):
        filtered_text = re.sub(r'[^a-zA-Z0-9 \n!@#$%^&*()_+\-=\[\]{};:"\\|,.<>\/?]', '', text)
        filtered_text = filtered_text.replace('.', '')
        return filtered_text.strip()

    @staticmethod
    def fix_slash_in_stats(stat):
        stat = stat.strip()

        if len(stat) == 3 and stat[1] == '1':
            return f"{stat[0]}/{stat[2]}"
        if len(stat) == 4 and stat[1] == '1':
            return f"{stat[0]}/{stat[2:]}"
        if len(stat) == 5 and stat[2] == '1':
            return f"{stat[0:2]}/{stat[3:]}"

        return stat

    def detect_text_in_image(self, image, region_name, allowlist=None):
        # Use CPU explicitly since we're in a CPU-only environment
        reader = easyocr.Reader(['en'], gpu=False)
    
        # Convert PIL image to OpenCV format
        img_cropped = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
        # Apply scaling and blurring only to numeric regions
        if "name" not in region_name and "grade" not in region_name:
            scale_factor = 2
            upscaled = cv2.resize(img_cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            blur = cv2.blur(upscaled, (5, 5))
            img_cropped = blur
    
        # Perform OCR on the processed image
        result = reader.readtext(img_cropped, detail=0, allowlist=allowlist, text_threshold=0.3)

        detected_texts = ['0' if text.strip() == '' else text for text in result]

        # Remove spaces from numeric stats
        if "name" not in region_name:
            detected_texts = [text.replace(' ', '') for text in detected_texts]
        return detected_texts

    def format_ocr_results(self, ocr_results, region_names):
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

                    # Validation checks
                    if stat_name == "grade" and not re.match(r'^[ABCDF][+-]?$|^$', text):
                        logging.warning(f"Invalid grade detected for player {player_number}: {text}")
                    elif stat_name != "name" and not text:
                        logging.warning(f"Empty value detected for {stat_name} of player {player_number}")

            elif "team1_q" in region_name:
                quarter_number = region_name[-1]
                formatted_output["teams"]["team1_quarters"][f"quarter_{quarter_number}"] = text if text else '0'

            elif "team2_q" in region_name:
                quarter_number = region_name[-1]
                formatted_output["teams"]["team2_quarters"][f"quarter_{quarter_number}"] = text if text else '0'

        return formatted_output

    def process_single_image(self, image_path, output_folder):
        """Process a single image and return results"""
        logging.info(f'Processing file: {image_path}')

        cropped_images, region_names = self.crop_and_save_regions(image_path)

        ocr_results = []
        for cropped_image, region_name in zip(cropped_images, region_names):
            allowlist = self.get_allowlist(region_name)
            ocr_text = self.detect_text_in_image(cropped_image, region_name, allowlist=allowlist)
            ocr_results.append(ocr_text)

        formatted_results = self.format_ocr_results(ocr_results, region_names)

        # Generate hash of the formatted results
        results_json_str = json.dumps(formatted_results, sort_keys=True)
        results_hash = hashlib.sha256(results_json_str.encode('utf-8')).hexdigest()
        formatted_results['hash'] = results_hash
        formatted_results['image_path'] = image_path
        formatted_results['mode'] = self.mode

        # Save results
        filename = os.path.basename(image_path)
        output_json_path = os.path.join(output_folder, f'{filename}_results.json')
        with open(output_json_path, 'w') as json_file:
            json.dump(formatted_results, json_file, indent=4)
        logging.info(f'Formatted results saved to {output_json_path}')

        return formatted_results

    def process_images(self, input_folder, output_folder):
        """Process all images in a folder"""
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(input_folder, filename)
                self.process_single_image(input_image_path, output_folder)

                # Move processed image to IMAGE_OUTPUT_FOLDER
                shutil.move(input_image_path, os.path.join(IMAGE_OUTPUT_FOLDER, filename))

    @staticmethod
    def correct_common_errors(text):
        corrections = {
            "Al Player": "AI Player",
            "Al Player 3": "AI Player",
        }
        return corrections.get(text, text)
    
    @staticmethod
    def get_team(player_number):
        if 1 <= player_number <= 5:
            return 'team1'
        elif 6 <= player_number <= 10:
            return 'team2'
        return 'unknown'
    
    @staticmethod
    def get_position(player_number):
        positions = {
            1: 'PG', 2: 'SG', 3: 'SF', 4: 'PF', 5: 'C',
            6: 'PG', 7: 'SG', 8: 'SF', 9: 'PF', 10: 'C'
        }
        return positions.get(player_number, '')

    @staticmethod
    def get_allowlist(region_name):
        if "FGMFGA" in region_name or "3PM3PA" in region_name or "FTMFTA" in region_name:
            return '0123456789/'
        if "grade" in region_name:
            return 'ABCDF+-'
        if "player" in region_name and "name" not in region_name:
            return '0123456789'
        return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-_/ '

def main():
    parser = argparse.ArgumentParser(description='NBA 2K Box Score OCR Processor')
    parser.add_argument('--input', '-i', default=IMAGE_INPUT_FOLDER, 
                       help='Input folder or image path')
    parser.add_argument('--output', '-o', default=JSON_OUTPUT_FOLDER,
                       help='Output folder for JSON results')
    parser.add_argument('--mode', '-m', choices=['yolo', 'legacy'], default='legacy',
                       help='Processing mode: yolo (YOLO detection) or legacy (fixed coordinates)')
    parser.add_argument('--yolo-model', default=YOLO_MODEL_PATH,
                       help='Path to YOLO model weights')
    
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

    ocr_processor = OCRProcessor(mode=args.mode, yolo_model_path=args.yolo_model)

    if os.path.isfile(args.input):
        # Process single image
        ocr_processor.process_single_image(args.input, args.output)
    else:
        # Process folder
        ocr_processor.process_images(args.input, args.output)

if __name__ == "__main__":
    main()