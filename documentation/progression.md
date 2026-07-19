# Project Progression: Deepfake Detection using ViT

This file tracks the current work done and leftover work for the Deepfake Detection project.

---

## Overall Progress Summary

- **Total Tasks:** 30
- **Completed:** 30
- **In Progress:** 0
- **Leftover (Not Started):** 0

---

## Step-by-Step Task Tracker

### Phase 1: Data Preparation & Preprocessing
- [x] **Step 1.1:** Parse the `metadata.json` files for DFDC Parts 0-6.
- [x] **Step 1.2:** Write a balancing script to isolate an equal count of REAL and FAKE video IDs.
- [x] **Step 1.3:** Develop the OpenCV video ingestion pipeline to sample 15 frames per target video.
- [x] **Step 1.4:** Implement MTCNN for facial detection on the sampled frames.
- [x] **Step 1.5:** Apply the bounding box expansion logic (margin addition) and save the cropped faces as `.jpg` files organized into `dataset/processed_faces_v2/train/REAL`, `dataset/processed_faces_v2/train/FAKE`, `dataset/processed_faces_v2/val/REAL`, and `dataset/processed_faces_v2/val/FAKE` directories based on the Video-ID split rule.

### Phase 2: Model Setup & Hyperparameter Tuning
- [x] **Step 2.1:** Initialize the Hugging Face `transformers` library and load the pre-trained base ViT.
- [x] **Step 2.2:** Construct the custom PyTorch `nn.Module` class, adding a Dropout layer and a final Linear layer (classification head) outputting a single binary logit.
- [x] **Step 2.3:** Configure the data loaders with image augmentations (RandomHorizontalFlip, ColorJitter) to improve generalization (Batch size set to 8 for 8GB Mac M2).
- [x] **Step 2.4:** Set up the PyTorch `mps` device mapping to utilize local hardware acceleration.
- [x] **Step 2.5:** Initialize the `AdamW` optimizer (learning rate: 3e-5, weight decay: 0.01) and configure a Linear Warmup with Cosine Decay scheduler.
 
### Phase 3: Training & Evaluation (Code Implemented with Gradient Accumulation)
- [x] **Step 3.1:** Execute the training loop (3 to 5 epochs). Monitor training loss vs. validation loss continuously to prevent overfitting (Optimized with Gradient Accumulation of 4 steps to achieve effective batch size 32 on 8GB hardware).
- [x] **Step 3.2:** Run inference on the holdout validation set.
- [x] **Step 3.3:** Calculate required evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- [x] **Step 3.4:** Save the optimal model weights to a `.pth` file.

### Phase 4: Application Development (Completed)
- [x] **Step 4.1:** Build the core Streamlit UI (titles, file uploaders, layout).
- [x] **Step 4.2:** Integrate the PyTorch inference loop into the Streamlit backend.
- [x] **Step 4.3:** Develop the logic for video uploads (extracting frames, running batched inference, aggregating the final confidence score).
- [x] **Step 4.4:** (Bonus) Implement Attention Rollout visualization to generate a heatmap indicating which parts of the face the ViT focused on to make its decision.

### Phase 5: Final Packaging & Consolidation
- [x] **Step 5.1:** Clean and document the source code (merged notebooks and standardized paths).
- [x] **Step 5.2:** Prepare the final Presentation and PDF Report mapping the workflow.
- [x] **Step 5.3:** Push all assets to a structured GitHub repository.

---

### Phase 6: AI-Generated Face Detection Extension (StyleGAN Dataset)

> **Context:** The existing ViT model was trained exclusively on DFDC (face-swap deepfakes) and therefore fails to detect AI-generated faces (e.g., StyleGAN, Midjourney, Stable Diffusion). This phase extends the model's detection capability to cover fully AI-synthesized faces using the **140k Real and Fake Faces** dataset (`dataset/Fake_face_detection_with_Keras/real_vs_fake/real-vs-fake/`).
>
> **Dataset profile:**
> - Source: FFHQ (real) + StyleGAN (fake)
> - Image size: 256×256 RGB, already face-cropped — no MTCNN needed
> - Train: 100,000 images (50K real, 50K fake — perfectly balanced)
> - Valid: 20,000 images (10K real, 10K fake)
> - Test: 20,000 images (10K real, 10K fake)
> - Folder structure: `train/real/`, `train/fake/`, `valid/real/`, `valid/fake/`, `test/real/`, `test/fake/` — compatible with PyTorch `ImageFolder`

- [x] **Step 6.1: Data Audit & Compatibility Verification**
  - Verify the StyleGAN dataset folder structure (`train/real`, `train/fake`, `valid/real`, `valid/fake`) is compatible with the existing `ImageFolder` pipeline used in Phase 1.
  - Confirm image dimensions (256×256 RGB) and ensure the ViT processor's resize transform will correctly **downscale** them to 224×224 (256→224 is a downscale, not an upscale — no MTCNN crop needed since images are pre-cropped).
  - Check for any corrupted or incomplete image files across all splits before training.

- [x] **Step 6.2: Consolidate Mixed-Dataset Fine-tuning**
  - Consolidate all StyleGAN fine-tuning cells directly into the main `dfd_ViT.ipynb` notebook to maintain a single cohesive, linear pipeline.
  - **Explicitly set the PyTorch device to `mps`** at the top of the notebook to ensure M2 GPU acceleration is used.
  - **Implement a mixed (concatenated) dataset** using `ConcatDataset` to train on both the original DFDC data and the new StyleGAN dataset simultaneously.
  - For the **validation split**, keep DFDC val and StyleGAN val as separate loaders so per-dataset metrics remain interpretable.
  - Apply the same augmentation pipeline (RandomHorizontalFlip, ColorJitter, Normalize) and validation transforms to both datasets.

- [x] **Step 6.3: Transfer Learning — Fine-tune from Existing Checkpoint**
  - Load the best existing DFDC checkpoint (`Model/best_vit_deepfake_2.pth`) as the starting weights instead of training from scratch.
  - Freeze the ViT backbone layers initially and only train the classification head for the first 2 epochs, then unfreeze all layers for full fine-tuning.

- [x] **Step 6.4: Hyperparameter Configuration for Fine-tuning**
  - Use a lower learning rate (e.g., `1e-5`) than the original training run.
  - Keep `weight_decay = 0.05` and `ACCUMULATION_STEPS = 4` from the v2 configuration.
  - Set `EPOCHS = 10` with `patience = 3` early stopping watching val loss.
  - Save new checkpoint to `Model/best_vit_deepfake_3.pth` to keep all weight files separate and safe.

- [x] **Step 6.5: Training & Evaluation**
  - Execute the fine-tuning loop, monitoring train loss vs. val loss for overfitting.
  - After training completes, run a full evaluation on the held-out `test/` split (20K images) to get final Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Compare test-set performance against the DFDC-only model to confirm generalization improvement.

- [x] **Step 6.6: Cross-Dataset Evaluation**
  - Run `Model/best_vit_deepfake_3.pth` on a sample of the original DFDC `val` set to verify it has not forgotten face-swap detection.
  - Run `Model/best_vit_deepfake_2.pth` on the StyleGAN `test` set to document the baseline failure rate (expected ~0% detection).
  - This cross-test confirms whether the fine-tuned model is truly dual-capability.

- [x] **Step 6.7: Dual-Model Strategy Decision**
  - Decided on strategy: **Option A (Single Unified Model)** selected as v3 retains 100.4% of v2's DFDC accuracy and achieves 99.25% accuracy on StyleGAN.

- [x] **Step 6.8: App Integration**
  - Update `app.py` to load the new model weight(s) based on the strategy chosen in Step 6.7.
  - Update the verdict card and heatmap explanation UI to mention that the system now detects both face-swap deepfakes and AI-generated (StyleGAN-type) faces.
  - Re-test the app manually with known real images, DFDC fakes, and StyleGAN fakes to confirm all three cases produce correct verdicts.

- [x] **Step 6.9: Notebook Comparison Cell**
  - Add a final cell in `dfd_ViT.ipynb` that runs a side-by-side evaluation of `Model/best_vit_deepfake_2.pth` vs `Model/best_vit_deepfake_3.pth` on both the DFDC val set and the StyleGAN test set, printing a formatted comparison table.

---

## Log of Completed Work

### 2026-07-19
- Consolidate all Phase 6 training, validation, evaluation, and side-by-side model comparison code directly into `dfd_ViT.ipynb`, and removed the separate `dfd_ViT_stylegan.ipynb` to clean up the repository.
- Reorganized all dataset directories to exist under a unified root-level `dataset/` folder.
- Successfully fine-tuned the model on a mixed dataset of DFDC and StyleGAN (v3 model).
- Validated performance:
  - **DFDC Val:** Accuracy 91.75% (+0.35% improvement over v2), ROC-AUC 0.9752.
  - **StyleGAN Test:** Accuracy 99.25% (+47.33% improvement over v2), ROC-AUC 0.9995.
- Created `Model/` directory and moved all 3 checkpoints into it, keeping git tracking intact.
- Updated `app.py` and `dfd_ViT_stylegan.ipynb` to refer to the new paths and updated `.gitattributes` for LFS tracking.

### 2026-07-12
- Started the project setup.
- Created `requirements.txt` listing all necessary libraries.
- Initiated installation of project dependencies (`torchvision`, `transformers`, `facenet-pytorch`, `streamlit`, `scikit-learn`, etc.).

