import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from src.data.dataset import MultimodalMedicalDataset
from src.models.multimodal import MultimodalMedicalReportGenerator

def get_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Medical Report Generator")
    
    # Paths
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv", help="Path to train CSV")
    parser.add_argument("--val-csv", type=str, default="data/splits/val.csv", help="Path to val CSV")
    parser.add_argument("--img-dir", type=str, default="data/raw/images", help="Path to images directory")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/vlm_experiment", help="TensorBoard log directory")
    
    # Model configuration
    parser.add_argument("--vision-model", type=str, default="google/vit-base-patch16-224", help="Vision encoder backbone")
    parser.add_argument("--language-model", type=str, default="gpt2", help="Language decoder backbone")
    parser.add_argument("--projector-type", type=str, default="mlp", choices=["linear", "mlp"], help="Projection layer type")
    
    # Training Stage
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], 
                        help="1: Projector Alignment (freeze vision & language), 2: LoRA Fine-Tuning (freeze vision, LoRA language)")
    
    # LoRA Hyperparameters (Only for Stage 2)
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max report token length")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Learning rate warmup steps")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 1. Initialize TensorBoard logging
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"Logging training telemetry to TensorBoard: {args.log_dir}")

    # 2. Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 3. Setup Tokenizer
    print(f"Loading Tokenizer: {args.language_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 4. Instantiate Datasets & Loaders
    print("Loading datasets...")
    train_dataset = MultimodalMedicalDataset(
        csv_file=args.train_csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        image_size=224,
        split="train",
        max_length=args.max_seq_len
    )
    val_dataset = MultimodalMedicalDataset(
        csv_file=args.val_csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        image_size=224,
        split="val",
        max_length=args.max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train Dataset Size: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val Dataset Size:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    # 5. Build Model based on the Training Stage
    print(f"Assembling VLM for Stage {args.stage}...")
    
    if args.stage == 1:
        # Stage 1: Freeze Vision Encoder & Language Decoder. Train only the Projector.
        model = MultimodalMedicalReportGenerator(
            vision_model=args.vision_model,
            language_model=args.language_model,
            projector_type=args.projector_type,
            freeze_vision=True,
            use_lora=False # Keep LLM completely frozen
        )
        
        # Verify frozen parameters in LLM
        for param in model.language_decoder.parameters():
            param.requires_grad = False
            
        print("Stage 1 verification: Vision and Language networks frozen. Projector is trainable.")
        
    elif args.stage == 2:
        # Stage 2: Freeze Vision Encoder. Apply LoRA and train Language Decoder + Projector.
        model = MultimodalMedicalReportGenerator(
            vision_model=args.vision_model,
            language_model=args.language_model,
            projector_type=args.projector_type,
            freeze_vision=True,
            use_lora=True, # Apply LoRA layers
            lora_r=args.lora_r
        )
        print("Stage 2 verification: Vision network frozen. Projector and Language LoRA parameters are trainable.")
        
    model = model.to(device)

    # 6. Setup Optimizer and Cosine Scheduler
    # Separate parameters so we only optimize those that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=num_training_steps
    )

    # 7. Setup AMP (Automatic Mixed Precision) for memory efficiency in Google Colab
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 8. Training Loop
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images, input_ids, attention_mask, labels)
                loss = outputs["loss"]
                
            # Backward pass & gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            loss_val = loss.item()
            total_train_loss += loss_val
            
            # Progress bar update
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            # Log step loss to TensorBoard
            global_step = epoch * len(train_loader) + step
            writer.add_scalar("Loss/Train_Step", loss_val, global_step)
            writer.add_scalar("LR/Step", scheduler.get_last_lr()[0], global_step)
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        # 9. Validation Loop
        model.eval()
        total_val_loss = 0
        print("Evaluating model...")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images, input_ids, attention_mask, labels)
                    loss = outputs["loss"]
                    
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Average Val Loss:   {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)

        # 10. Checkpoint Saving
        if (epoch + 1) % args.save_every == 0 or avg_val_loss < best_val_loss:
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"New best validation loss achieved: {best_val_loss:.4f}")
            
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "stage": args.stage,
                "args": vars(args)
            }
            
            # Save periodic checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"vlm_stage{args.stage}_epoch{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best checkpoint
            if is_best:
                best_path = os.path.join(args.output_dir, f"vlm_stage{args.stage}_best.pt")
                torch.save(checkpoint, best_path)
                print(f"Saved best model weights to: {best_path}")
                
    writer.close()
    print("Training run successfully finished!")

if __name__ == "__main__":
    main()
