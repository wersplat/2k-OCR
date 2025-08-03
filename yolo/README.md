# NBA 2K Box Score YOLOv8 Detection

This module integrates YOLOv8 object detection with the existing EasyOCR pipeline to dynamically detect and extract stats from NBA 2K box score screenshots.

## Features

- Replaces fixed cropping logic with dynamic object detection
- Detects 12 stat fields per player (player name, grade, points, etc.)
- Supports 10 players per screen (5 per team)
- Includes fallback to fixed regions when detection fails
- Visualizes detections with different colors per stat type
- Exports structured JSON output compatible with existing pipeline

## Directory Structure

```
yolo/
├── data/
│   ├── dataset.yaml       # Dataset configuration
│   ├── images/            # Training/validation images
│   └── labels/            # YOLO format labels
├── models/                # Trained models are saved here
├── scripts/
│   ├── train.py           # Training script
│   └── detect.py          # Detection script
├── LABELING_GUIDE.md      # Instructions for labeling data
├── README.md              # This file
├── main.py                # Main entry point
└── yolo_ocr_processor.py  # Core integration module
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Collect NBA 2K box score screenshots
2. Label the data using makesense.ai or Roboflow following the instructions in `LABELING_GUIDE.md`
3. Organize the data in the YOLO format:
   - `data/images/train/` - Training images
   - `data/images/val/` - Validation images
   - `data/labels/train/` - Training labels
   - `data/labels/val/` - Validation labels

### Training

Train a YOLOv8 model on your labeled data:

```bash
python yolo/main.py train --data yolo/data/dataset.yaml --epochs 100 --model-size n --pretrained
```

Options:
- `--data`: Path to dataset configuration file
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--img-size`: Image size for training
- `--model-size`: YOLOv8 model size (n, s, m, l, x)
- `--pretrained`: Use pretrained weights
- `--device`: Device to use (cuda device or cpu)
- `--project`: Project directory for saving results
- `--name`: Experiment name

### Detection

Process images using the trained model:

```bash
python yolo/main.py detect --model yolo/models/2k_stats_detector/weights/best.pt --source ./toProcess/images/ --visualize
```

Options:
- `--model`: Path to trained YOLOv8 model
- `--source`: Path to image or directory of images
- `--conf`: Confidence threshold
- `--output`: Output directory for JSON results
- `--visualize`: Visualize detections
- `--vis-output`: Output directory for visualizations
- `--no-fallback`: Disable fallback to fixed regions

### Export

Export the trained model to different formats:

```bash
python yolo/main.py export --model yolo/models/2k_stats_detector/weights/best.pt --format onnx
```

Options:
- `--model`: Path to trained YOLOv8 model
- `--format`: Export format (onnx, torchscript, openvino, coreml)
- `--output`: Output directory for exported model

## Integration with Existing Pipeline

The YOLOv8 OCR processor can be used as a drop-in replacement for the existing OCR processor:

```python
from yolo.yolo_ocr_processor import YOLOOCRProcessor

# Initialize the processor
processor = YOLOOCRProcessor(
    model_path='yolo/models/2k_stats_detector/weights/best.pt',
    conf_threshold=0.25,
    use_fallback=True,
    visualize=True
)

# Process a single image
results = processor.process_image('path/to/image.jpg', 'output/folder')

# Process multiple images
processor.process_images('input/folder', 'output/folder', 'visualizations/folder')
```

## Fallback Mechanism

If the YOLOv8 model fails to detect a sufficient number of stat regions, the processor will automatically fall back to the fixed cropping logic from the original OCR processor. This ensures robustness even when the model encounters unfamiliar screenshots or overlays.

## Visualization

When the `visualize` option is enabled, the processor will generate visualization images with bounding boxes around detected stat regions. Each stat type is color-coded for easy identification:

- Player name: Green
- Grade: Blue
- Points: Red
- Rebounds: Cyan
- Assists: Magenta
- Steals: Yellow
- Blocks: Dark Blue
- Fouls: Dark Green
- Turnovers: Dark Red
- FGM/FGA: Dark Cyan
- 3PM/3PA: Dark Magenta
- FTM/FTA: Dark Yellow
