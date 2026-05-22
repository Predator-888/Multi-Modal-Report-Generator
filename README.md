---
title: Multimodal Medical Report Generator
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.40.0
app_file: demo/app.py
pinned: false
---

# Multimodal Medical Report Generator
An end-to-end Vision-Language Model (VLM) for automated chest X-ray screening.

## Features
- **Image-to-Text Generation**: Converts chest X-ray images into detailed radiology reports.
- **Pre-trained Models**: Built on state-of-the-art architectures (ResNet, BERT, GPT-2).
- **User-Friendly Interface**: Interactive Gradio demo for easy testing.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers
- Gradio

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Multi-Modal-Report-Generator
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision transformers gradio
   ```

## Usage

### Run the Demo
Start the Gradio interface:
```bash
python demo/app.py
```

### Training
Train the model on your custom dataset:
```bash
python src/train.py \
    --image_dir path/to/images \
    --report_path path/to/reports.csv \
    --output_dir ./checkpoints \
    --epochs 10
```

### Inference
Generate reports for new images:
```python
from src.models.multimodal import MultimodalVLM
from src.utils.preprocess import preprocess_image
from transformers import AutoTokenizer

# Load model
model = MultimodalVLM.from_pretrained("local/path/to/checkpoint")

# Preprocess image
image = preprocess_image("path/to/image.png")

# Generate report
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
report = model.generate(
    images=image,
    tokenizer=tokenizer,
    prompt_text="Generate a radiology report for this chest X-ray."
)

print(report)
```

## Model Architecture

The model consists of three main components:
1. **Image Encoder**: Pre-trained ResNet-50 to extract visual features.
2. **Language Model**: Pre-trained GPT-2 (or BERT) for text generation.
3. **Fusion Layer**: Combines visual and textual features for conditional generation.

## Dataset

The default dataset used for pre-training is **MIMIC-CXR**. For custom datasets:
- Ensure reports are in CSV format with `image_id` and `report` columns.
- Preprocess images to a consistent size (e.g., 224x224).

## License

MIT License
