import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(image_size=224, split="train"):
    """
    Get albumentations transforms for X-ray images.
    
    CRITICAL MEDICAL AI RULE:
    Do NOT apply horizontal flipping (Flip / HorizontalFlip) to chest X-rays.
    In clinical settings, left-right orientation is critical (e.g., detecting dextrocardia, 
    matching the side of pleural effusion or pneumothorax). Flipping the image changes 
    the anatomy and renders the generated report factually incorrect.
    """
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            # Enhance low-contrast structures in lung fields
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),
            # Small affine transformations to simulate positioning variations
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
            # Brightness and contrast shifts
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            # Standard normalization for vision models
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # For validation and testing, apply only deterministic transformations
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # Apply CLAHE to match train distribution
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
