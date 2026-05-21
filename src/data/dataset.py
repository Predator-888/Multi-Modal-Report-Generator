import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from src.data.augmentations import get_transforms

class MultimodalMedicalDataset(Dataset):
    """
    PyTorch Dataset for pairing Chest X-Ray images with radiology reports.
    Compatible with auto-regressive decoders (e.g. BioGPT, GPT-2, Mistral).
    """
    def __init__(self, csv_file, img_dir, tokenizer, image_size=224, split="train", max_length=128):
        """
        Args:
            csv_file (str): Path to the split CSV file (train/val/test).
            img_dir (str): Path to the directory containing images.
            tokenizer: Pre-trained HuggingFace tokenizer.
            image_size (int): Image dimension to resize to.
            split (str): "train", "val", or "test" to apply appropriate augmentations.
            max_length (int): Maximum sequence length for the text tokens.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = get_transforms(image_size=image_size, split=split)
        self.max_length = max_length

        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token: {self.tokenizer.eos_token}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        text_report = str(row["text_report"])

        # Load X-ray image
        img_path = os.path.join(self.img_dir, image_name)
        if not os.path.exists(img_path):
            # Fallback in case of missing images: create a dummy black image
            print(f"Warning: Image not found at {img_path}. Using a dummy black image.")
            image = Image.new("RGB", (224, 224), color=0)
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Using a dummy black image.")
                image = Image.new("RGB", (224, 224), color=0)

        # Apply image transformations
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        image_tensor = augmented["image"]

        # Tokenize report
        tokenized = self.tokenizer(
            text_report,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # [SeqLen]
        attention_mask = tokenized["attention_mask"].squeeze(0)  # [SeqLen]

        # For autoregressive training, labels are the same as input_ids.
        # But we must mask the padding tokens so they do not contribute to the loss.
        # Standard PyTorch CrossEntropyLoss ignores index -100.
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "image": image_tensor,          # [3, H, W]
            "input_ids": input_ids,          # [SeqLen]
            "attention_mask": attention_mask,  # [SeqLen]
            "labels": labels                # [SeqLen]
        }
