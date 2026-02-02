# YouTube Viral Predictor (yt-viral-predictor)

A deep learning model that predicts the virality potential of YouTube videos based on their **thumbnail** and **title** - the two key elements viewers see before clicking.

## Overview

This project uses a **multimodal neural network** combining:
- **ResNet50** for visual feature extraction from thumbnails
- **DistilBERT** for text feature extraction from titles
- **MLP classifier** for final viral/non-viral prediction

### V-Score Metric

Videos are labeled using a custom **Logarithmic V-Score** — a normalized metric measuring how well a video performs compared to the channel's historical baseline:

```
V-Score = (log(views + 1) - baseline_median) / baseline_std
```

| V-Score | Interpretation |
|---------|----------------|
| > 1.0 |  Viral potential |
| -0.5 to 1.0 |  Average |
| < -0.5 |  Underperforming |

### Data Split

The dataset is split into:
- **90% Training set** - used for model training
- **10% Test set** - used for evaluation (stratified to maintain class balance)

## Project Structure

```
yt-viral-predictor/
├── main.py           # Data collection script
├── download.py       # YouTube API + yt-dlp downloader
├── dataset.py        # PyTorch Dataset class
├── model.py          # ViralPredictor model architecture
├── train.py          # Training script
├── api.txt           # Paste your YouTube API key here (not committed)
├── requirements.txt  # Python dependencies
├── data/
│   ├── raw/          # CSV files + thumbnails
│   └── models/       # Trained model weights
└── docs/             # Documentation
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up YouTube API key

```bash
echo "YOUR_YOUTUBE_API_KEY" > api.txt
```

### 3. Collect data

```bash
python main.py
```

### 4. Train the model

```bash
python train.py
```

The trained model will be saved to `data/models/viral_predictor.pth`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers (HuggingFace)
- yt-dlp
- google-api-python-client

## Model Architecture

```
Thumbnail (224×224×3)          Title (max 50 tokens)
        │                              │
        ▼                              ▼
   ResNet50                      DistilBERT
   [2048 dim]                    [768 dim]
   (frozen)                       (frozen)
        │                              │
        └──────────┬───────────────────┘
                   │
            Concatenate [2816]
                   │
                   ▼
            MLP Classifier
            Linear(1024) → BatchNorm → ReLU → Dropout(0.5)
            Linear(256) → BatchNorm → ReLU → Dropout(0.3)
            Linear(1) → Sigmoid
                   │
                   ▼
           Viral Probability
```

### Training Features
- **Frozen backbone**: ResNet50 and DistilBERT layers are frozen (only classifier trains)
- **Data Augmentation**: RandomCrop, ColorJitter, RandomRotation, RandomErasing
- **Early Stopping**: Based on test accuracy with patience=10
- **Learning Rate Scheduler**: ReduceLROnPlateau

### Hyperparameters
- Batch size: 16
- Learning rate: 3e-4 (classifier only)
- Weight decay: 1e-3
- Train/Test split: 90/10

## License

Apache License
