# Project 4: Deepfake Detection using Vision Transformer (ViT)
**Project Documentation & Implementation Roadmap**

---

## 1. Project Overview (What We Are Building)
This project focuses on building an AI-powered system capable of distinguishing between real human faces and manipulated (deepfake) faces. Addressing the increasing realism of deepfake videos, the system will leverage a state-of-the-art Vision Transformer (ViT) architecture. Instead of processing full video frames (which include irrelevant background data), the pipeline will dynamically isolate facial features and analyze spatial anomalies and blending artifacts to determine authenticity. 

## 2. The End Result
The final deliverable will be a complete, end-to-end machine learning pipeline wrapped in a user-friendly Streamlit web application. 

**Core Application Features:**
* **Media Upload:** Support for both single images (`.jpg`, `.png`) and video files (`.mp4`).
* **Real-time Processing:** Automated facial extraction from the uploaded media.
* **Binary Prediction:** Clear classification output indicating if the media is **REAL** or **FAKE**.
* **Confidence Score:** A percentage-based metric displaying the model's certainty.
* **Video-Level Analytics:** For video uploads, the system will extract multiple frames across the timeline, analyze each face individually, and aggregate the predictions to provide a final video-level verdict.

**Academic Deliverables:**
* Source Code & GitHub Repository.
* Trained Model Weights (`.pth` file).
* Deployed Streamlit App.
* Final Presentation & PDF Report detailing problem statement, flow, and evaluation metrics.

---

## 3. Methodology & Architecture (How We Will Achieve It)

### Tech Stack & Hardware
* **Core Logic:** Python, PyTorch, OpenCV, Transformers (Hugging Face).
* **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks).
* **Model Architecture:** Pre-trained Vision Transformer (`google/vit-base-patch16-224-in21k`).
* **Frontend UI:** Streamlit.
* **Hardware Acceleration:** Apple Silicon (M2 via PyTorch `mps` backend for optimized local training and inference).

### The Implementation Strategy (Transfer Learning)
Rather than training a Vision Transformer entirely from scratch (which requires millions of images and risks heavy overfitting on local hardware), the system will utilize Transfer Learning. We will load a ViT pre-trained on generic image data, replace its final classification head, and fine-tune it specifically on the DFDC dataset. 

### Critical Data Engineering Rules
To ensure the model learns actual deepfake artifacts rather than taking statistical shortcuts, three strict data rules will be enforced:
1.  **1:1 Class Balancing:** The heavy bias of the DFDC dataset (majority fake) will be programmatically corrected to ensure exactly equal numbers of REAL and FAKE videos are processed.
2.  **20-30% Bounding Box Margin:** Deepfake blending artifacts typically occur at the edges of the face (jawline, forehead). MTCNN will be configured to expand its facial crop by 20% to capture these outer seams.
3.  **Video-ID Splitting:** To prevent "Actor Leakage" (the model memorizing specific actors' faces), the Train/Validation split will be conducted at the video level, not the frame level.

---

## 4. Elaborate Step-by-Step Roadmap

### Phase 1: Data Preparation & Preprocessing
* **Step 1.1:** Parse the `metadata.json` files for DFDC Parts 0-4.
* **Step 1.2:** Write a balancing script to isolate an equal count of REAL and FAKE video IDs.
* **Step 1.3:** Develop the OpenCV video ingestion pipeline to sample 5 to 10 frames per target video.
* **Step 1.4:** Implement MTCNN for facial detection on the sampled frames.
* **Step 1.5:** Apply the bounding box expansion logic (margin addition) and save the cropped faces as `.jpg` files organized into `train/REAL`, `train/FAKE`, `val/REAL`, and `val/FAKE` directories based on the Video-ID split rule.

### Phase 2: Model Setup & Hyperparameter Tuning
* **Step 2.1:** Initialize the Hugging Face `transformers` library and load the pre-trained base ViT.
* **Step 2.2:** Construct the custom PyTorch `nn.Module` class, adding a Dropout layer and a final Linear layer (classification head) outputting a single binary logit.
* **Step 2.3:** Configure the data loaders with image augmentations (RandomHorizontalFlip, ColorJitter) to improve generalization.
* **Step 2.4:** Set up the PyTorch `mps` device mapping to utilize local hardware acceleration.
* **Step 2.5:** Initialize the `AdamW` optimizer (learning rate: 3e-5, weight decay: 0.01) and configure a Linear Warmup with Cosine Decay scheduler.

### Phase 3: Training & Evaluation
* **Step 3.1:** Execute the training loop (3 to 5 epochs). Monitor training loss vs. validation loss continuously to prevent overfitting.
* **Step 3.2:** Run inference on the holdout validation set.
* **Step 3.3:** Calculate required evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* **Step 3.4:** Save the optimal model weights to a `.pth` file.

### Phase 4: Application Development
* **Step 4.1:** Build the core Streamlit UI (titles, file uploaders, layout).
* **Step 4.2:** Integrate the PyTorch inference loop into the Streamlit backend.
* **Step 4.3:** Develop the logic for video uploads (extracting frames, running batched inference, aggregating the final confidence score).
* **Step 4.4:** (Bonus) Implement Attention Rollout visualization to generate a heatmap indicating which parts of the face the ViT focused on to make its decision.

### Phase 5: Final Packaging
* **Step 5.1:** Clean and document the source code.
* **Step 5.2:** Prepare the final Presentation and PDF Report mapping the workflow.
* **Step 5.3:** Push all assets to a structured GitHub repository.

---

## 5. Detailed Timeline & Estimated Hours (4-Day Sprint)

### Day 1: The Data Pipeline
* **Task 1:** Parse metadata and write the 1:1 class balancing algorithm. *(1.0 Hour)*
* **Task 2:** Write and test the OpenCV + MTCNN extraction script (including the 20% margin logic). *(2.0 Hours)*
* **Task 3:** Execute the extraction script across Parts 0-4 and verify directory structures/Video-ID splits. *(2.5 Hours - mostly automated processing time)*
* **Total Estimated Time:** 5.5 Hours

### Day 2: Architecture & Setup
* **Task 1:** Load pre-trained ViT and construct the custom PyTorch classification class. *(1.5 Hours)*
* **Task 2:** Setup PyTorch DataLoaders with necessary image augmentations. *(1.0 Hour)*
* **Task 3:** Configure hardware backend, AdamW optimizer, and Cosine Decay Scheduler. *(1.0 Hour)*
* **Total Estimated Time:** 3.5 Hours

### Day 3: The Heavy Training Grind
* **Task 1:** Write the formal PyTorch training and validation loop. *(1.5 Hours)*
* **Task 2:** Execute training runs. Monitor metrics, adjust learning rate if loss plateaus, and save the best model weights. *(4.0 to 5.0 Hours - active monitoring + compute)*
* **Task 3:** Generate evaluation metrics (Precision, Recall, F1, ROC-AUC) on the final validation set. *(1.0 Hour)*
* **Total Estimated Time:** 6.5 - 7.5 Hours

### Day 4: Streamlit Deployment & Polish
* **Task 1:** Build the Streamlit interface and link the model inference pipeline for static images. *(2.0 Hours)*
* **Task 2:** Implement the video frame extraction and aggregation logic for video uploads. *(1.5 Hours)*
* **Task 3:** (Optional Bonus) Integrate Grad-CAM/Attention maps into the UI output. *(1.5 Hours)*
* **Task 4:** Finalize documentation, PDF report, and GitHub repository upload. *(2.0 Hours)*
* **Total Estimated Time:** 7.0 Hours

**Total Project Effort:** ~22.5 to 23.5 Hours
