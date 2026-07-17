# Project Progression: Deepfake Detection using ViT

This file tracks the current work done and leftover work for the Deepfake Detection project.

---

## Overall Progress Summary

- **Total Tasks:** 21
- **Completed:** 18
- **In Progress:** 0
- **Leftover (Not Started):** 3

---

## Step-by-Step Task Tracker

### Phase 1: Data Preparation & Preprocessing
- [x] **Step 1.1:** Parse the `metadata.json` files for DFDC Parts 0-6.
- [x] **Step 1.2:** Write a balancing script to isolate an equal count of REAL and FAKE video IDs.
- [x] **Step 1.3:** Develop the OpenCV video ingestion pipeline to sample 15 frames per target video.
- [x] **Step 1.4:** Implement MTCNN for facial detection on the sampled frames.
- [x] **Step 1.5:** Apply the bounding box expansion logic (margin addition) and save the cropped faces as `.jpg` files organized into `train/REAL`, `train/FAKE`, `val/REAL`, and `val/FAKE` directories based on the Video-ID split rule.

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

### Phase 5: Final Packaging
- [ ] **Step 5.1:** Clean and document the source code.
- [ ] **Step 5.2:** Prepare the final Presentation and PDF Report mapping the workflow.
- [ ] **Step 5.3:** Push all assets to a structured GitHub repository.

---

## Log of Completed Work

### 2026-07-12
- Started the project setup.
- Created `requirements.txt` listing all necessary libraries.
- Initiated installation of project dependencies (`torchvision`, `transformers`, `facenet-pytorch`, `streamlit`, `scikit-learn`, etc.).
