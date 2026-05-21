import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

class MedicalVisionEncoder(nn.Module):
    """
    Wrapper for pre-trained vision encoders.
    Extracts dense, patch-level spatial embeddings from chest X-rays.
    Supports ViT architectures and specialized medical backbones.
    """
    def __init__(self, model_name="google/vit-base-patch16-224", freeze=True):
        super().__init__()
        print(f"Initializing Vision Encoder: {model_name}")
        
        # Load backbone
        if "vit" in model_name.lower():
            self.backbone = ViTModel.from_pretrained(model_name)
            self.vision_dim = self.backbone.config.hidden_size
            self.is_vit = True
        else:
            # Fallback for general models (e.g. ResNet backbones or BioViL-T)
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            # Try to infer dimension, fallback to 768
            self.vision_dim = getattr(self.backbone.config, "hidden_size", 768)
            self.is_vit = False

        # Freeze weights if requested (common in Stage 1 alignment)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("Vision encoder weights successfully frozen.")
        else:
            self.backbone.train()

    def forward(self, pixel_values):
        """
        Args:
            pixel_values (torch.Tensor): Visual input tensor [B, 3, H, W]
        Returns:
            torch.Tensor: Patch embeddings [B, NumPatches, VisionDim]
        """
        if self.is_vit:
            outputs = self.backbone(pixel_values)
            # last_hidden_state contains [B, NumPatches + 1 (CLS), VisionDim]
            # We return all patch embeddings including the CLS token
            return outputs.last_hidden_state
        else:
            # For non-ViT encoders, we obtain spatial features and flatten spatial dimensions
            outputs = self.backbone(pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            elif hasattr(outputs, "pooler_output"):
                # If only pooler output exists, add a dummy patch dimension: [B, 1, VisionDim]
                return outputs.pooler_output.unsqueeze(1)
            else:
                # Direct tensor output from custom models
                features = outputs[0] if isinstance(outputs, tuple) else outputs
                if len(features.shape) == 4: # [B, C, H, W]
                    B, C, H, W = features.shape
                    # Flatten to [B, H*W, C]
                    return features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                return features
