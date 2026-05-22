import torch
import torch.nn as nn
from src.models.vision_encoder import MedicalVisionEncoder
from src.models.projector import MultimodalProjector
from src.models.language_decoder import MedicalLanguageDecoder

class MultimodalMedicalReportGenerator(nn.Module):
    """
    End-to-end Vision-Language Model (VLM) for radiology report generation.
    Connects a MedicalVisionEncoder with a MedicalLanguageDecoder using a MultimodalProjector.
    """
    def __init__(self, vision_model="google/vit-base-patch16-224", 
                 language_model="gpt2", projector_type="mlp",
                 freeze_vision=True, use_lora=False, lora_r=8):
        super().__init__()
        
        # 1. Vision Encoder
        self.vision_encoder = MedicalVisionEncoder(model_name=vision_model, freeze=freeze_vision)
        
        # 2. Language Decoder
        self.language_decoder = MedicalLanguageDecoder(
            model_name=language_model, 
            use_lora=use_lora, 
            lora_r=lora_r
        )
        
        # 3. Multimodal Projector
        self.projector = MultimodalProjector(
            vision_dim=self.vision_encoder.vision_dim,
            text_dim=self.language_decoder.text_dim,
            projector_type=projector_type
        )

    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Training forward pass. Concatenates visual patch embeddings with report token embeddings.
        
        Args:
            images (torch.Tensor): Preprocessed image batch [B, 3, H, W]
            input_ids (torch.Tensor): Report token IDs [B, SeqLen]
            attention_mask (torch.Tensor): Report attention mask [B, SeqLen]
            labels (torch.Tensor, optional): Report target tokens [B, SeqLen]
        Returns:
            dict: Loss and logits
        """
        B = images.size(0)

        # 1. Extract visual features
        # Shape: [B, NumPatches, VisionDim]
        vision_feats = self.vision_encoder(images)

        # 2. Project visual features to language model dimension
        # Shape: [B, NumPatches, TextDim]
        projected_visual_embeds = self.projector(vision_feats)
        num_patches = projected_visual_embeds.size(1)

        # 3. Get token embeddings from language decoder
        # Get standard input embedding layer
        decoder_model = self.language_decoder.get_decoder()
        if hasattr(decoder_model, "get_input_embeddings"):
            embedding_layer = decoder_model.get_input_embeddings()
        else:
            embedding_layer = decoder_model.base_model.get_input_embeddings()

        # Shape: [B, SeqLen, TextDim]
        text_embeds = embedding_layer(input_ids)

        # 4. Concatenate projected visual embeddings with text embeddings
        # Combined sequence length: NumPatches + SeqLen
        # Shape: [B, NumPatches + SeqLen, TextDim]
        combined_embeds = torch.cat([projected_visual_embeds, text_embeds], dim=1)

        # 5. Expand attention mask to cover visual patches (always attended to, set mask to 1)
        vis_mask = torch.ones((B, num_patches), dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # 6. Adjust labels for training (fill -100 for visual patch positions to ignore in cross-entropy loss)
        combined_labels = None
        if labels is not None:
            vis_labels = torch.full((B, num_patches), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([vis_labels, labels], dim=1)

        # 7. Pass to language decoder (cast inputs_embeds to match decoder's parameter dtype)
        target_dtype = next(self.language_decoder.parameters()).dtype
        combined_embeds = combined_embeds.to(dtype=target_dtype)
        
        outputs = self.language_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }

    @torch.no_grad()
    def generate(self, images, tokenizer, max_new_tokens=64, temperature=0.7, top_k=50, top_p=0.9, prompt_text="", repetition_penalty=1.2, no_repeat_ngram_size=3):
        """
        Autoregressive generation script for report generation from image inputs.
        
        Args:
            images (torch.Tensor): Input X-ray image batch [B, 3, H, W]
            tokenizer: Tokenizer for vocabulary decoding
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Sampling temperature
            prompt_text (str): Optional text prompt prefix
        Returns:
            list: List of generated report strings
        """
        self.eval()
        B = images.size(0)

        # Extract and project visual embeddings
        vision_feats = self.vision_encoder(images)
        projected_visual_embeds = self.projector(vision_feats)
        num_patches = projected_visual_embeds.size(1)

        # Handle prompt prefix
        decoder_model = self.language_decoder.get_decoder()
        if hasattr(decoder_model, "get_input_embeddings"):
            embedding_layer = decoder_model.get_input_embeddings()
        else:
            embedding_layer = decoder_model.base_model.get_input_embeddings()

        if prompt_text:
            tokens = tokenizer(prompt_text, return_tensors="pt").to(images.device)
            prompt_embeds = embedding_layer(tokens["input_ids"]).repeat(B, 1, 1)
            inputs_embeds = torch.cat([projected_visual_embeds, prompt_embeds], dim=1)
            
            # Setup attention mask
            prompt_mask = tokens["attention_mask"].repeat(B, 1)
            vis_mask = torch.ones((B, num_patches), dtype=prompt_mask.dtype, device=images.device)
            attention_mask = torch.cat([vis_mask, prompt_mask], dim=1)
        else:
            inputs_embeds = projected_visual_embeds
            attention_mask = torch.ones((B, num_patches), dtype=torch.long, device=images.device)

        # Call HF generate on embeddings
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature if temperature > 0.0 else None,
            "top_k": top_k if temperature > 0.0 else None,
            "top_p": top_p if temperature > 0.0 else None,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        # Generate token sequences (cast inputs_embeds to match decoder model's parameter dtype)
        target_dtype = next(decoder_model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)
        
        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **{k: v for k, v in generation_config.items() if v is not None}
        )

        # Decode tokens to natural text
        decoded_outputs = []
        for g_ids in generated_ids:
            decoded_outputs.append(tokenizer.decode(g_ids, skip_special_tokens=True))
            
        return decoded_outputs
