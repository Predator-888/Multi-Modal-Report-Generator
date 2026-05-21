import os
import sys
# Dynamically add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

from src.data.augmentations import get_transforms
from src.models.multimodal import MultimodalMedicalReportGenerator

def get_args(args=None):
    parser = argparse.ArgumentParser(description="Run single-image report generation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image file (PNG/JPEG)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint .pt file")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--prompt", type=str, default="Findings:", help="Prompt prefix for report generation")
    return parser.parse_args(args)

def main(args=None):
    args = get_args(args)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Load Checkpoint Metadata
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint_data["args"]

    # 3. Initialize Tokenizer
    lang_model = ckpt_args["language_model"]
    print(f"Loading Tokenizer: {lang_model}")
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Process Input Image
    print(f"Preprocessing image: {args.image}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at: {args.image}")
    
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    
    # Load test transforms (CLAHE dynamic enhancement)
    transform = get_transforms(image_size=224, split="val")
    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device) # Add batch dimension: [1, 3, H, W]

    # 5. Build and Load Model
    print("Assembling VLM architecture...")
    model = MultimodalMedicalReportGenerator(
        vision_model=ckpt_args["vision_model"],
        language_model=ckpt_args["language_model"],
        projector_type=ckpt_args["projector_type"],
        freeze_vision=True,
        use_lora=(checkpoint_data.get("stage", 1) == 2),
        lora_r=ckpt_args.get("lora_r", 8)
    )
    
    # Load weights
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Weights loaded successfully!")

    # 6. Autoregressive Generation
    print("\nGenerating report...")
    generated_reports = model.generate(
        images=image_tensor,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_text=args.prompt
    )

    print("\n" + "=" * 40)
    print("             GENERATED REPORT           ")
    print("=" * 40)
    print(f"{args.prompt} {generated_reports[0].strip()}")
    print("=" * 40)

if __name__ == "__main__":
    main()
