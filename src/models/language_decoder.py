import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

class MedicalLanguageDecoder(nn.Module):
    """
    Wrapper for causal language decoders (e.g., GPT-2, BioGPT, Mistral-7B).
    Handles base model loading, weight freezing, and LoRA adapter integration.
    """
    def __init__(self, model_name="gpt2", use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        print(f"Initializing Language Decoder: {model_name}")
        
        # Determine device-specific configurations (e.g., flash attention or bfloat16 if running on Colab A100/T4)
        device_map = None
        torch_dtype = torch.float32
        
        if torch.cuda.is_available():
            device_map = "auto"
            # Use float16 or bfloat16 for efficient training on GPU
            torch_dtype = torch.float16

        # Load standard configuration and model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Load decoder model
        # Note: In a free Colab environment, Mistral-7B must be loaded in 8-bit or 4-bit (via bitsandbytes).
        # We write standard loading here, letting accelerate and device_map handle memory placement.
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device_map if "mistral" in model_name.lower() else None,
            trust_remote_code=True
        )
        
        # Get hidden dimension (embedding dimension) of LLM
        self.text_dim = getattr(self.config, "n_embd", None) or getattr(self.config, "hidden_size", 768)
        
        # Integrate LoRA if requested (Stage 2 fine-tuning)
        self.use_lora = use_lora
        if use_lora:
            print("Applying PEFT LoRA adapters to language decoder...")
            
            # Auto-configure target modules based on model architecture
            target_modules = ["c_attn"] if "gpt2" in model_name.lower() else ["q_proj", "v_proj"]
            if "biogpt" in model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            self.decoder = get_peft_model(self.decoder, peft_config)
            self.decoder.print_trainable_parameters()

    def get_decoder(self):
        """Return the underlying model or PEFT model wrapper."""
        return self.decoder

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        """
        Args:
            inputs_embeds (torch.Tensor): Continuous embeddings fed to transformer blocks [B, SeqLen, TextDim]
            attention_mask (torch.Tensor, optional): Standard causal attention mask [B, SeqLen]
            labels (torch.Tensor, optional): Target token IDs for shifted cross-entropy [B, SeqLen]
        Returns:
            CausalLMOutputWithPast: HuggingFace model outputs containing loss and logits.
        """
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
