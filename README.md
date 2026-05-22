---
title: Multimodal Medical Report Generator
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.40.0
app_file: demo/app.py
pinned: false
python_version: 3.10
---

# Multimodal Medical Report Generator
An end-to-end Vision-Language Model (VLM) for automated chest X-ray screening.

## Features
- **Image-to-Text Generation**: Converts chest X-ray images into detailed radiology reports.
- **Pre-trained Models**: Built on state-of-the-art architectures (ViT, GPT-2, LoRA PEFT).
- **User-Friendly Interface**: Interactive Gradio demo for easy testing.

## Getting Started

### Prerequisites
- Python 3.10
- PyTorch
- Transformers
- Gradio

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gauravpa8459/multimodal-medical-report-generator
   cd Multi-Modal-Report-Generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Demo
Start the Gradio interface:
```bash
python demo/app.py
```

## Model Architecture
The model consists of three main components:
1. **Vision Encoder**: Pre-trained ViT-Base (`google/vit-base-patch16-224`) to extract spatial patch visual features.
2. **Language Decoder**: Causal GPT-2 model adapted with PEFT LoRA adapters.
3. **Multimodal Projector**: MLP projection layer mapping visual features into the text token embedding space.

## License
MIT License