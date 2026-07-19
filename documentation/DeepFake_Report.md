# Project: Deepfake & Synthetic Face Detection using Vision Transformer (ViT)
**Project Documentation, Architecture, and Implementation Report**

---

## 1. Project Overview
This project builds an AI-powered system capable of distinguishing between real human faces and manipulated (deepfake) or synthesized faces. Addressing the increasing realism of deepfake videos and AI-synthesized profiles (GANs), the system leverages a pre-trained Vision Transformer (ViT) architecture. Instead of processing full video frames (which include irrelevant background data), the pipeline dynamically isolates facial crops and analyzes spatial anomalies and blending artifacts to determine authenticity.

## 2. The Final Deliverable
The final deliverable is a complete, end-to-end machine learning pipeline wrapped in a user-friendly Streamlit web application. 

**Core Application Features:**
* **Media Upload:** Support for both single images (`.jpg`, `.png`) and video files (`.mp4`).
* **Real-time Processing:** Automated facial extraction using MTCNN.
* **Dual Verdicts:** Identifies both Face-Swap Deepfakes (DFDC) and Synthetic GAN Faces (StyleGAN).
* **Explainable AI (XAI):** Generates an Attention Rollout heatmap projecting ViT attention weights back onto the facial image, visually highlighting the features that influenced the model's decision.
* **Video-Level Analytics:** For video uploads, the system samples frames across the timeline, runs batched inference, and aggregates predictions to provide a final video-level verdict.

---

## 3. Methodology & Architecture

### Tech Stack & Hardware
* **Core Logic:** Python, PyTorch, OpenCV, Transformers (Hugging Face).
* **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks).
* **Model Architecture:** Pre-trained Vision Transformer (`google/vit-base-patch16-224-in21k`).
* **Frontend UI:** Streamlit.
* **Hardware Acceleration:** Apple Silicon (M2 via PyTorch `mps` backend for optimized local training and inference).

### Data Reorganization
All datasets are organized inside a unified `dataset/` root directory to keep the repository clean:
* `dataset/processed_faces_v2/` — MTCNN pre-processed face crops from the DFDC dataset.
* `dataset/Fake_face_detection_with_Keras/` — 140k Real and Fake faces (FFHQ + StyleGAN).

### The Implementation Strategy (Mixed Fine-Tuning)
To prevent **Catastrophic Forgetting** (where training on a new distribution causes a model to lose previous knowledge), we implemented a mixed-training approach:
1. **Initial Baseline (v2)**: Fine-tuned on DFDC face-swap datasets to detect surgical manipulation boundaries.
2. **Transfer Learning (v3)**: Loaded `Model/best_vit_deepfake_2.pth`, froze the ViT backbone, and warmed up the classification head on a balanced mixture of DFDC and StyleGAN data using `ConcatDataset`.
3. **Full Fine-Tuning**: Unfroze all layers of the ViT backbone and trained with a lower learning rate (1e-5) to optimize classification boundaries across both domains.

---

## 4. Performance & Evaluation

Below is the side-by-side comparison of the baseline model `v2` (DFDC-only) vs the final model `v3` (Mixed fine-tuned):

### DFDC Validation Set
| Metric | v2 (DFDC-only) | v3 (Mixed fine-tuned) | Delta | Status |
|---|---|---|---|---|
| **Accuracy** | 91.39% | **91.75%** | +0.35% | ✅ *No Forgetting* |
| **Precision** | 91.44% | **91.82%** | +0.38% | ✅ |
| **Recall** | 91.66% | **91.97%** | +0.31% | ✅ |
| **F1-Score** | 91.55% | **91.90%** | +0.35% | ✅ |
| **ROC-AUC** | 0.9738 | **0.9752** | +0.0014 | ✅ |

### StyleGAN Test Set
| Metric | v2 (DFDC-only) | v3 (Mixed fine-tuned) | Delta | Status |
|---|---|---|---|---|
| **Accuracy** | 51.92% | **99.25%** | +47.33% | ✅ *Dual-Capability* |
| **Precision** | 51.14% | **99.38%** | +48.24% | ✅ |
| **Recall** | 86.43% | **99.11%** | +12.68% | ✅ |
| **F1-Score** | 64.26% | **99.24%** | +34.98% | ✅ |
| **ROC-AUC** | 0.4958 | **0.9995** | +0.5037 | ✅ |

### Key Observations
* **Catastrophic Forgetting Prevented**: By using `ConcatDataset` instead of sequential training, the model retained 100.4% of its face-swap detection capability on DFDC.
* **FFHQ Feature Synergies**: The addition of high-quality FFHQ real images during the fine-tuning phase actually *improved* DFDC metrics marginally.
* **Superb StyleGAN Generalization**: The StyleGAN detection accuracy went from a coin flip (51.92%) to near-perfect (99.25%) with a ROC-AUC of 0.9995.

---

## 5. End-to-End Execution Roadmap

### Phase 1: Data Ingestion & Preprocessing
* Parse `metadata.json` files and apply 1:1 class balancing for DFDC videos.
* Extract equidistant frames and run CPU-bound MTCNN face detection with a 30% margin expansion to capture jawline/forehead seams.
* Save crops into train/val folders.

### Phase 2: Base Model Architecture Setup
* Load pre-trained ViT (`google/vit-base-patch16-224-in21k`).
* Build a custom PyTorch wrapper with a Dropout layer (0.2) and a single-logit linear classification head.

### Phase 3: Baseline Training (DFDC)
* Configure AdamW optimizer with learning rate 3e-5.
* Use Gradient Accumulation (4 steps) to simulate batch size 32 on local 8GB memory limits.
* Save baseline checkpoint to `Model/best_vit_deepfake_2.pth`.

### Phase 4: Mixed Dataset & Extension Training (StyleGAN)
* Subsample the StyleGAN dataset to match DFDC size (~38k images) to maintain distribution balance.
* Concatenate training datasets using `ConcatDataset` while keeping validation loaders separate.
* Load `Model/best_vit_deepfake_2.pth`, freeze ViT backbone, and warm up the head for 2 epochs.
* Unfreeze backbone and fine-tune for 8 epochs with learning rate 1e-5.
* Save final checkpoints to `Model/best_vit_deepfake_3.pth`.

### Phase 5: Streamlit Integration & Explainability (XAI)
* Connect Streamlit uploader to the MTCNN detector and the `Model/best_vit_deepfake_3.pth` inference pipeline.
* Implement the Attention Rollout algorithm across all 12 attention heads/layers to project self-attention maps onto the face crops.
