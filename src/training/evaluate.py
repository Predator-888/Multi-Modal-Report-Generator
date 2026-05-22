import os
import sys
# Dynamically add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import pandas as pd

from src.data.dataset import MultimodalMedicalDataset
from src.models.multimodal import MultimodalMedicalReportGenerator

# Ensure NLTK packages are ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def get_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate VLM Report Generation performance")
    
    # Paths
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv", help="Path to test CSV split")
    parser.add_argument("--img-dir", type=str, default="data/raw/images", help="Path to images directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/metrics", help="Directory to save evaluation results")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated report length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    
    return parser.parse_args(args)

def calculate_metrics(hypotheses, references):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4, and ROUGE-L metrics.
    
    Args:
        hypotheses (list of str): Generated reports.
        references (list of str): Target ground-truth reports.
    Returns:
        dict: Calculated NLP metric averages.
    """
    print("Calculating generation metrics...")
    
    # Tokenize sentences into words for NLTK BLEU score
    tokenized_hyps = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]
    tokenized_refs = [[nltk.word_tokenize(ref.lower())] for ref in references] # BLEU expects a list of reference lists
    
    smooth = SmoothingFunction().method1
    
    # 1. Calculate BLEU Scores
    bleu1 = corpus_bleu(tokenized_refs, tokenized_hyps, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(tokenized_refs, tokenized_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(tokenized_refs, tokenized_hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(tokenized_refs, tokenized_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # 2. Calculate ROUGE Scores
    # We will fall back to a lightweight character/word LCS implementation or import rouge_score if available
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(references, hypotheses)]
        avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
    except ImportError:
        print("rouge-score package not found. Using simple word-level ROUGE-L approximation...")
        # Custom word-level ROUGE-L approximation using longest common subsequence
        def lcs(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]
            
        approx_scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.lower().split()
            hyp_words = hyp.lower().split()
            if not ref_words or not hyp_words:
                approx_scores.append(0.0)
                continue
            lcs_len = lcs(ref_words, hyp_words)
            prec = lcs_len / len(hyp_words)
            rec = lcs_len / len(ref_words)
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            approx_scores.append(f1)
        avg_rouge_l = sum(approx_scores) / len(approx_scores)

    return {
        "BLEU-1": round(bleu1, 4),
        "BLEU-2": round(bleu2, 4),
        "BLEU-3": round(bleu3, 4),
        "BLEU-4": round(bleu4, 4),
        "ROUGE-L": round(avg_rouge_l, 4)
    }

def main(args=None):
    args = get_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Checkpoint Metadata
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint_data["args"]
    
    # 2. Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # 3. Load Tokenizer
    lang_model = ckpt_args["language_model"]
    print(f"Loading Tokenizer: {lang_model}")
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Dataset
    print("Loading test dataset splits...")
    test_dataset = MultimodalMedicalDataset(
        csv_file=args.test_csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        image_size=224,
        split="test",
        max_length=ckpt_args["max_seq_len"]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2
    )

    # 5. Initialize Model
    print("Building VLM architecture...")
    model = MultimodalMedicalReportGenerator(
        vision_model=ckpt_args["vision_model"],
        language_model=ckpt_args["language_model"],
        projector_type=ckpt_args["projector_type"],
        freeze_vision=True,
        use_lora=(checkpoint_data.get("stage", 1) == 2), # Apply LoRA wrapper if Stage 2
        lora_r=ckpt_args.get("lora_r", 8)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model weights successfully loaded into architecture!")

    # 6. Autoregressive Inference Loop
    hypotheses = []
    references = []
    sample_showcases = []

    print("\nStarting report generation on held-out test split...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"]
            
            # Generate predictions using our continuous-embed prompt
            preds = model.generate(
                images=images,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                prompt_text="Findings:"
            )
            
            # Decode references (ground truth)
            for idx, pred in enumerate(preds):
                gt = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                
                # Clean up prefixes if they are generated or padded
                cleaned_pred = pred.strip()
                cleaned_gt = gt.strip()
                
                hypotheses.append(cleaned_pred)
                references.append(cleaned_gt)
                
                # Keep a few showcase samples for visualization
                if len(sample_showcases) < 5:
                    sample_showcases.append({
                        "Generated": cleaned_pred,
                        "Ground_Truth": cleaned_gt
                    })

    # 7. Compute Final Metrics
    scores = calculate_metrics(hypotheses, references)
    
    print("\n================ EVALUATION SUMMARY ================")
    for metric, score in scores.items():
        print(f" * {metric:<10}: {score:.4f}")
    print("====================================================")

    # 8. Save results to disk
    results = {
        "checkpoint": args.checkpoint,
        "metrics": scores,
        "sample_outputs": sample_showcases
    }
    
    out_file = os.path.join(args.output_dir, "test_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation scores and sample outputs saved to: {out_file}")

    # Generate a detailed predictions CSV
    df_preds = pd.DataFrame({
        "Ground_Truth": references,
        "Generated_Prediction": hypotheses
    })
    csv_out = os.path.join(args.output_dir, "test_predictions.csv")
    df_preds.to_csv(csv_out, index=False)
    print(f"Complete predictions spreadsheet saved to: {csv_out}")

if __name__ == "__main__":
    main()
