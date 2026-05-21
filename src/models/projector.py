import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    """
    Projection layer to align visual features with language decoder token space.
    Supports either a simple Linear projector or a non-linear MLP (Linear -> GELU -> Linear).
    """
    def __init__(self, vision_dim=768, text_dim=768, projector_type="mlp"):
        super().__init__()
        print(f"Initializing Multimodal Projector (Type: {projector_type}, {vision_dim} -> {text_dim})")
        
        if projector_type == "linear":
            self.projector = nn.Linear(vision_dim, text_dim)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, text_dim),
                nn.GELU(),
                nn.Linear(text_dim, text_dim)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual patch embeddings [B, NumPatches, VisionDim]
        Returns:
            torch.Tensor: Projected features aligned with text embedding space [B, NumPatches, TextDim]
        """
        return self.projector(x)
