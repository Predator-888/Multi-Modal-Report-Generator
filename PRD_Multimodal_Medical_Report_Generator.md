# 📋 Product Requirements Document (PRD)
# Multimodal Medical Report Generator

---

> **Version:** 1.0  
> **Type:** Research-Grade Solo Project  
> **Estimated Duration:** 3–4 Months  
> **Domain:** Healthcare / Bio-AI + Computer Vision + NLP  
> **Author:** BTech AI & ML Student  

---

## 1. Project Overview

### 1.1 Summary
The **Multimodal Medical Report Generator** is an AI system that accepts a **chest X-ray image** as input and automatically generates a **clinically accurate radiology report** in natural language. It mimics the workflow of a radiologist — analyzing visual findings in the image and translating them into structured medical text.

### 1.2 Problem Statement
- Radiologists globally are overburdened, leading to delayed diagnoses.
- In developing countries like India, there is a severe shortage of radiologists (~1 per 100,000 patients in rural areas).
- Manual report generation is time-consuming, inconsistent, and prone to fatigue-induced errors.
- An AI-assisted report generator can serve as a **first-pass screening tool**, reducing workload and improving turnaround time.

### 1.3 Proposed Solution
A Vision-Language Model (VLM) pipeline that:
1. Encodes the X-ray image using a **medical vision encoder**
2. Passes visual embeddings into a **language decoder**
3. Generates a structured radiology report covering findings, impressions, and recommendations

---

## 2. Goals & Objectives

### 2.1 Primary Goals
- Build a working end-to-end multimodal pipeline (image → text)
- Fine-tune a pre-trained vision-language model on medical imaging data
- Achieve competitive performance on standard clinical NLP metrics
- Create a deployable demo interface

### 2.2 Research Goals
- Explore cross-modal attention between visual patches and clinical text tokens
- Study the effect of domain-adaptive pretraining vs. direct fine-tuning
- Optionally contribute findings as a workshop paper (IEEE EMBC / ACL BioNLP)

### 2.3 Out of Scope
- Diagnosing conditions beyond chest X-ray (CT, MRI out of scope for v1)
- Real clinical deployment or patient data handling
- Multi-disease co-occurrence scoring (stretch goal for v2)

---

## 3. Tech Stack

### 3.1 Core Framework

| Layer | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.11 | Primary development language |
| Deep Learning | PyTorch 2.x | Model training & fine-tuning |
| Vision Encoder | BioViL-T / CheXagent | Chest X-ray specific visual features |
| Language Decoder | BioGPT / Mistral-7B-Instruct | Medical text generation |
| Multimodal Bridge | LLaVA architecture (custom) | Connecting vision + language |
| Training Utilities | HuggingFace Transformers + PEFT | Fine-tuning with LoRA adapters |
| Data Processing | Albumentations, Pillow, Pandas | Image augmentation + data wrangling |
| Evaluation | NLTK, evaluate (HuggingFace) | BLEU, ROUGE, CIDEr scoring |
| Experiment Tracking | Weights & Biases (WandB) | Loss curves, metric dashboards |
| Demo Interface | Gradio | Interactive web demo |
| Version Control | Git + GitHub | Code management |
| Environment | Conda / venv | Dependency management |

### 3.2 Hardware Requirements

| Option | Specs | Notes |
|---|---|---|
| Minimum | 8GB VRAM GPU (RTX 3060) | Use LoRA, 8-bit quantization |
| Recommended | 16GB VRAM (RTX 4080 / A100) | Full fine-tuning possible |
| Free Cloud Options | Google Colab Pro / Kaggle | Sufficient for prototyping |
| Best Free Option | Kaggle (30hr/week T4 x2) | Recommended for students |

### 3.3 Key Python Packages
```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0                  # LoRA fine-tuning
accelerate>=0.28.0            # Multi-GPU / mixed precision
datasets>=2.18.0              # HuggingFace datasets
evaluate>=0.4.0               # Metrics (BLEU, ROUGE)
albumentations>=1.3.0         # Image augmentation
gradio>=4.0.0                 # Demo UI
wandb>=0.17.0                 # Experiment tracking
Pillow>=10.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
```

---

## 4. Datasets

### 4.1 Primary Dataset — MIMIC-CXR
- **Source:** PhysioNet (free academic access)
- **Size:** ~227,000 chest X-rays with paired radiology reports
- **Access:** Requires credentialing at physionet.org (free, takes ~1–2 days)
- **Format:** DICOM images + structured text reports
- **Use:** Main training and evaluation dataset

### 4.2 Secondary Dataset — CheXpert
- **Source:** Stanford ML Group
- **Size:** 224,316 chest X-rays, 14 labeled pathologies
- **Access:** Free registration at stanfordmlgroup.github.io
- **Use:** Pre-training the vision encoder for pathology recognition

### 4.3 Evaluation Dataset — IU X-Ray (Indiana University)
- **Source:** Open-i NIH
- **Size:** 7,470 X-rays with reports (smaller, great for quick eval)
- **Access:** Publicly available, no registration needed
- **Use:** Held-out test set for final evaluation

### 4.4 Data Preprocessing Pipeline
```
Raw DICOM → Normalize (0-1) → Resize (224×224 or 512×512)
→ CLAHE Enhancement → Albumentations Augmentation
→ Tokenize Reports (BioTokenizer) → Dataset splits (80/10/10)
```

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│         Chest X-Ray Image (224×224 RGB)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 VISION ENCODER                          │
│    BioViL-T / CheXagent (ViT-B/16 backbone)            │
│    Output: Visual patch embeddings [N × 768]            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            MULTIMODAL PROJECTION LAYER                  │
│     Linear / MLP Projector (Visual → Text space)        │
│     Aligns vision embeddings with LLM token space       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               LANGUAGE DECODER                          │
│     BioGPT-Large / Mistral-7B (LoRA fine-tuned)         │
│     Cross-attention over visual + text tokens           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                          │
│         Generated Radiology Report                      │
│   ├── Findings (detailed visual observations)           │
│   ├── Impression (clinical summary)                     │
│   └── Recommendation (follow-up if needed)             │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Training Strategy

| Stage | What happens | Duration |
|---|---|---|
| Stage 1 | Freeze LLM, train only projector layer | Week 5–6 |
| Stage 2 | Unfreeze LLM, apply LoRA adapters, full fine-tune | Week 7–10 |
| Stage 3 | RLHF-lite: use clinical feedback signals for reward | Week 11–12 (stretch) |

---

## 6. Evaluation Metrics

### 6.1 NLP Metrics
| Metric | Target Score | What it measures |
|---|---|---|
| BLEU-4 | > 0.15 | N-gram overlap with ground truth |
| ROUGE-L | > 0.35 | Longest common subsequence |
| CIDEr | > 0.40 | Consensus-based image description |
| BERTScore | > 0.85 | Semantic similarity (preferred for medical) |

### 6.2 Clinical Metrics
| Metric | Tool | What it measures |
|---|---|---|
| CheXpert labeler accuracy | CheXpert NLP tool | Are correct pathologies mentioned? |
| Factual consistency score | AlignScore | Are generated facts grounded in the image? |
| Clinical correctness (manual) | Self-review with radiologist checklist | Human evaluation on 100 samples |

---

## 7. Project Milestones & Timeline

### Month 1 — Foundation
| Week | Task |
|---|---|
| Week 1 | Setup environment, access MIMIC-CXR & CheXpert datasets |
| Week 2 | EDA (Exploratory Data Analysis) — understand report structure, pathology distribution |
| Week 3 | Build data preprocessing pipeline (DICOM → tensor, report tokenization) |
| Week 4 | Implement baseline: Simple ViT encoder + GPT-2 decoder (sanity check) |

### Month 2 — Model Development
| Week | Task |
|---|---|
| Week 5 | Integrate BioViL-T vision encoder, implement projection layer |
| Week 6 | Stage 1 training — train only projection layer, evaluate on IU X-Ray |
| Week 7 | Integrate BioGPT / Mistral-7B with LoRA, begin Stage 2 fine-tuning |
| Week 8 | Hyperparameter tuning, WandB experiment tracking, ablation studies |

### Month 3 — Evaluation & Refinement
| Week | Task |
|---|---|
| Week 9 | Full evaluation on MIMIC-CXR test split (BLEU, ROUGE, BERTScore) |
| Week 10 | CheXpert labeler evaluation, error analysis, model improvements |
| Week 11 | Build Gradio demo interface, write inference pipeline |
| Week 12 | Final testing, edge case handling, performance benchmarking |

### Month 4 — Polish & Publish
| Week | Task |
|---|---|
| Week 13 | Write clean GitHub README with results, architecture diagram, demo GIF |
| Week 14 | Write a 4-page technical report / workshop paper draft |
| Week 15 | Record demo video, deploy Gradio app on HuggingFace Spaces |
| Week 16 | Submit to IEEE EMBC Student Abstract / ACL BioNLP Workshop (optional) |

---

## 8. Repository Structure

```
multimodal-medical-report-gen/
│
├── data/
│   ├── raw/                    # Raw DICOM files (not pushed to git)
│   ├── processed/              # Preprocessed tensors & tokenized reports
│   └── splits/                 # train/val/test CSV splits
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── preprocess.py       # DICOM → tensor pipeline
│   │   └── augmentations.py    # Albumentations transforms
│   │
│   ├── models/
│   │   ├── vision_encoder.py   # BioViL-T wrapper
│   │   ├── projector.py        # MLP projection layer
│   │   ├── language_decoder.py # BioGPT / Mistral wrapper
│   │   └── multimodal.py       # Full pipeline assembly
│   │
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Metric computation
│   │   └── config.yaml         # Hyperparameters
│   │
│   └── inference/
│       ├── predict.py          # Single image inference
│       └── batch_predict.py    # Batch evaluation
│
├── demo/
│   └── app.py                  # Gradio interface
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_results_analysis.ipynb
│
├── results/
│   ├── metrics/                # Saved evaluation scores
│   └── sample_outputs/         # Example generated reports
│
├── requirements.txt
├── README.md                   # Full project documentation
└── .gitignore
```

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MIMIC-CXR access delay | Medium | High | Start with IU X-Ray (no registration) while waiting |
| GPU memory constraints | High | High | Use LoRA + 8-bit quantization + gradient checkpointing |
| Model generates hallucinated findings | High | High | Add factual consistency scoring as a training signal |
| Overfitting on small medical vocab | Medium | Medium | Use domain-adaptive pretraining, data augmentation |
| Slow training on free GPU tiers | High | Medium | Use Kaggle dual T4, mixed precision (fp16) training |

---

## 10. Success Criteria

The project is considered **complete and portfolio-ready** when:

- [ ] End-to-end pipeline works (X-ray in → report out)
- [ ] BLEU-4 > 0.15 and BERTScore > 0.85 on MIMIC-CXR test split
- [ ] CheXpert labeler F1 > 0.70 on pathology mentions
- [ ] Gradio demo is live on HuggingFace Spaces
- [ ] GitHub repo has clean README, architecture diagram, and sample outputs
- [ ] Comparison table against at least 2 baseline models documented
- [ ] (Stretch) Workshop paper submitted or posted on arXiv

---

## 11. Resources & References

### Papers to Read First (in order)
1. **CheXNet** — Rajpurkar et al., 2017 — Foundation for chest X-ray AI
2. **BioViL** — Bannur et al., 2022 — Vision-language for radiology
3. **LLaVA** — Liu et al., 2023 — Multimodal instruction tuning architecture
4. **CheXagent** — Chen et al., 2024 — Most recent chest X-ray foundation model
5. **R2Gen** — Chen et al., 2020 — Radiology report generation benchmark

### Useful Links
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- IU X-Ray: https://openi.nlm.nih.gov/
- BioViL HuggingFace: https://huggingface.co/microsoft/BioViL-T
- BioGPT: https://huggingface.co/microsoft/biogpt
- CheXagent: https://huggingface.co/StanfordAIMI/CheXagent

---

*PRD Version 1.0 — Multimodal Medical Report Generator*
*Next steps: Set up environment → Access datasets → Begin EDA (Week 1)*
