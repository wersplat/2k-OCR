# NBA 2K OCR - End-to-End Box Score Processing System

A comprehensive OCR system for extracting player and team statistics from NBA 2K box score screenshots. This system combines YOLOv8 for stat region detection, EasyOCR for text recognition, and provides a complete workflow from image processing to model training.

## 🏀 Features

### Core OCR Processing

- **Dual Mode Processing**: Support for both legacy (fixed coordinates) and YOLO-based region detection
- **Multi-format Support**: Handles PNG, JPG, JPEG images
- **Structured Output**: Generates JSON with player stats, team quarters, and metadata
- **Batch Processing**: Process single images or entire folders

### Web Dashboard

- **Modern UI**: Clean, responsive dashboard built with FastAPI and Bootstrap
- **Real-time Processing**: Upload and process images through the web interface
- **Result Visualization**: View OCR results side-by-side with original images
- **Reprocessing**: Re-run OCR with different modes and settings
- **Status Monitoring**: Track processed images and system status

### Label Studio Integration

- **Automated Task Generation**: Convert OCR results to Label Studio tasks
- **Pre-filled Annotations**: Bounding boxes and labels automatically generated
- **YOLO Export**: Export annotations for model training
- **Docker Integration**: Ready-to-use Label Studio container

### YOLO Training Pipeline

- **Custom Model Training**: Train YOLOv8 models for stat region detection
- **Dataset Management**: Automatic dataset splitting and validation
- **Model Export**: Export to ONNX, TorchScript, or TFLite formats
- **Performance Monitoring**: Track mAP, precision, and recall metrics

## 📁 Project Structure

```
2k-OCR/
├── automate_2k.py              # Main OCR processing script
├── dashboard/                  # Web dashboard
│   ├── main.py                # FastAPI application
│   ├── templates/             # HTML templates
│   └── Dockerfile             # Dashboard container
├── yolo/                      # YOLO training and models
│   ├── train_yolo.py          # Training script
│   ├── data/                  # Dataset directory
│   └── models/                # Trained models
├── labelstudio_task_gen.py    # Label Studio task generator
├── processed/                 # Processed results
│   ├── images/               # Processed images
│   └── json/                 # OCR results
├── toProcess/                # Input directory
│   └── images/               # Images to process
├── labelstudio_tasks/        # Generated Label Studio tasks
├── docker-compose.yml        # Container orchestration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized setup)
- CUDA-capable GPU (optional, for faster processing)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd 2k-OCR
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**

   ```bash
   mkdir -p processed/images processed/json toProcess/images
   ```

### Basic Usage

#### Process Images via CLI

```bash
# Process a single image
python automate_2k.py --input path/to/image.jpg --mode legacy

# Process all images in a folder
python automate_2k.py --input ./toProcess/images --mode yolo

# Use custom YOLO model
python automate_2k.py --input ./toProcess/images --mode yolo --yolo-model ./yolo/models/best.pt
```

#### Start Web Dashboard

```bash
# Run locally
python dashboard/main.py

# Or use Docker Compose
docker-compose up dashboard
```

#### Generate Label Studio Tasks

```bash
# Generate tasks from processed results
python labelstudio_task_gen.py --export-yolo

# Export for YOLO training
python labelstudio_task_gen.py --export-yolo --yolo-output ./yolo/data
```

#### Train YOLO Model

```bash
# Train with default settings
python yolo/train_yolo.py --data-dir ./yolo/data --epochs 100

# Train with custom parameters
python yolo/train_yolo.py --data-dir ./yolo/data --model-size m --epochs 200 --batch-size 32
```

## 🐳 Docker Setup

### Full System with Docker Compose

```bash
# Start all services
docker-compose up -d

# Access services
# Dashboard: http://localhost:8000
# Label Studio: http://localhost:8080
```

### Individual Services

```bash
# Start only Label Studio
docker-compose up labelstudio

# Start only dashboard
docker-compose up dashboard
```

## 📊 OCR Output Format

The system generates structured JSON output:

```json
{
  "players": [
    {
      "player_number": 1,
      "position": "PG",
      "team": "team1",
      "name": "Player Name",
      "grade": "A+",
      "points": "25",
      "rebounds": "5",
      "assists": "8",
      "steals": "2",
      "blocks": "0",
      "fouls": "1",
      "tos": "2",
      "FGM": "10",
      "FGA": "18",
      "3PM": "3",
      "3PA": "7",
      "FTM": "2",
      "FTA": "2"
    }
  ],
  "teams": {
    "team1_quarters": {
      "quarter_1": "28",
      "quarter_2": "32",
      "quarter_3": "25",
      "quarter_4": "30"
    },
    "team2_quarters": {
      "quarter_1": "25",
      "quarter_2": "30",
      "quarter_3": "28",
      "quarter_4": "35"
    }
  },
  "hash": "abc123...",
  "image_path": "/path/to/image.jpg",
  "mode": "legacy"
}
```

## 🎯 YOLO Training

### Dataset Preparation

1. Process images with OCR to generate initial results
2. Generate Label Studio tasks: `python labelstudio_task_gen.py --export-yolo`
3. Review and correct annotations in Label Studio
4. Export corrected annotations back to YOLO format

### Training Configuration

```bash
# Basic training
python yolo/train_yolo.py --data-dir ./yolo/data --epochs 100

# Advanced training
python yolo/train_yolo.py \
  --data-dir ./yolo/data \
  --model-size m \
  --epochs 200 \
  --batch-size 32 \
  --split-dataset
```

### Model Classes

The system detects 13 different stat regions:

- `name`, `grade`, `points`, `rebounds`, `assists`, `steals`
- `blocks`, `fouls`, `tos`, `FGMFGA`, `3PM3PA`, `FTMFTA`, `team_quarter`

## 🔧 Configuration

### Environment Variables

```bash
# Dashboard settings
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000

# Label Studio settings
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/data
```

### Custom Coordinates

Modify the coordinate constants in `automate_2k.py` for different screen resolutions:

```python
BASE_X_PLAYER = 1219
PLAYER_Y_COORDINATES = [520, 602, 683, 765, 843, 1148, 1233, 1318, 1398, 1479]
```

## 📈 Performance Optimization

### GPU Acceleration

- Install CUDA for GPU-accelerated YOLO training
- Use `--device 0` for specific GPU selection
- Enable mixed precision training with `--half`

### Batch Processing

- Adjust batch size based on available memory
- Use `--batch-size 32` for larger models
- Monitor GPU memory usage during training

## 🐛 Troubleshooting

### Common Issues

**OCR Accuracy Issues**

- Ensure images are high quality (1920x1080 or higher)
- Check image orientation and lighting
- Try different OCR modes (legacy vs YOLO)

**YOLO Training Issues**

- Verify dataset structure matches expected format
- Check class labels in dataset.yaml
- Monitor training loss and validation metrics

**Docker Issues**

- Ensure ports 8000 and 8080 are available
- Check volume mount permissions
- Verify Docker and Docker Compose versions

### Logs and Debugging

```bash
# View dashboard logs
docker-compose logs dashboard

# View Label Studio logs
docker-compose logs labelstudio

# Check OCR processing logs
python automate_2k.py --input test.jpg --mode legacy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [Label Studio](https://github.com/heartexlabs/label-studio) for annotation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For issues and questions:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details
