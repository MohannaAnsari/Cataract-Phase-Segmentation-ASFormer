# Temporal Segmentation Paradigm

## Objective
The goal of this phase was to evaluate whether a **temporal segmentation model** could outperform the Phase 2 clip-level classification baseline in recognizing surgical phases.
While the previous I3D-based model predicted isolated clips independently, this phase focuses on **dense frame-level prediction** using a temporal transformer (ASFormer-lite) that explicitly models temporal continuity across the entire surgery.

---

## Methodology

### 1️⃣ Feature Extraction
To ensure efficient training on the GTX 1660 Ti GPU, a **pretrained ResNet-18** backbone (ImageNet weights) was used as a frozen frame-level feature extractor.
Each surgical video was processed frame by frame:
- Resize: 224 × 224  
- Normalize: ImageNet statistics  
- Sample every 2 frames (`FRAME_STEP = 2`)  
- Extracted 512-D features per frame  

This produced compact, semantically rich embeddings `[T, 512]` for each video.

---

### 2️⃣ Label Generation
Frame-wise ground-truth labels were reconstructed from the original annotation files (`annotations.csv`, `phases.csv`, `videos.csv`).
Each frame was assigned its corresponding `PhaseID − 1`.  
Train/validation splits followed an 80 / 20 division by video ID.

---

### 3️⃣ Model Architecture – ASFormer-lite
A lightweight **ASFormer (Action Segment Transformer)** was implemented in PyTorch:

| Component | Description |
|:-----------|:-------------|
| Input projection | 1×1 Conv (512 → 256) |
| Transformer encoder | 2 layers, 4 heads |
| Dilated residual blocks | Dilations = 1, 2, 4 |
| Output head | 1×1 Conv → 10 classes |

This architecture balances accuracy and computational efficiency while capturing long-range temporal dependencies.

---

### 4️⃣ Training Setup
| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 1 (per video) |
| Epochs | 25 |
| Loss | Cross-Entropy |
| Framework | PyTorch 2 + CUDA 11 |

Training was stable with smooth convergence.

---

## Results

### Quantitative Performance
| Metric | Validation Set |
|:--------|:----------------|
| **Mean Frame Accuracy** | **0.966** |
| **Macro F1-Score** | **0.651** |

The model achieved strong frame-wise accuracy, correctly predicting most phases.  
The lower F1 score indicates difficulty on rare or short transition phases, a common issue in imbalanced temporal datasets.

---

### Training Summary
| Epoch Range | Validation Acc | Validation Loss |
|--------------|----------------|-----------------|
| 1 – 5 | 0.95 – 0.96 | 0.18 → 0.15 |
| 10 – 15 | 0.95 – 0.96 | 0.15 → 0.12 |
| 20 – 25 | 0.96 – 0.97 | Stable |

Convergence was fast and consistent.

---

### Qualitative Visualization
![Phase Timeline Prediction](./outputs/plots/phase1_plot.png)

The **blue** line shows ground truth; the **orange** line shows predictions.  
The ASFormer captures the global structure well but occasionally produces short prediction spikes near phase transitions.

---

## Discussion

Compared with the Phase 2 I3D clip classifier (~ 0.30 accuracy), the ASFormer-lite model shows a **dramatic improvement** in both accuracy and temporal stability.

- **Temporal consistency:** Long-range memory smooths predictions and reduces spurious phase jumps.  
- **Efficiency:** The lightweight model runs comfortably on 6 GB VRAM.  
- **Remaining issue:** F1 < 0.7 due to class imbalance and rare short phases.  
- **Feature quality:** Pretrained ResNet-18 features yielded high performance without overfitting.

---

## Comparison with Phase 2

| Model | Approach | Feature Source | Accuracy | F1 | Notes |
|:------|:----------|:---------------|:----------|:----|:------|
| Phase 2 – I3D | Clip Classification | Learned from scratch | 0.30 | ~ 0.25 | Poor boundary recognition |
| **Phase 3 – ASFormer-lite** | Temporal Segmentation | Pretrained ResNet-18 | **0.97** | **0.65** | Robust temporal modeling |

---

## Conclusion
The experiment confirms that **temporal modeling is essential** for accurate surgical phase recognition.  
ASFormer-lite achieved > 3× accuracy improvement over the clip baseline while remaining computationally efficient — a promising result for real-time clinical systems.

---

## Future Work
Future work will explore integrating a **GraphCut-based temporal smoothing stage** to refine phase boundaries and enforce label consistency across frames.  
This approach draws on the author’s related research, *“
Graph-Cut-Based Semantic Optimization for Temporal Action Segmentation,”* which has been **accepted for presentation at the ICCKE 2025 Conference**.  
Incorporating GraphCut as a post-processing layer could reduce short-term misclassifications, improve temporal coherence, and connect this work with the author’s broader research on semantic consistency in video understanding.
