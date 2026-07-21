# 🎭 Vision Transformer (ViT) Deepfake & Synthetic Face Detector

> An end-to-end, explainable AI system utilizing Vision Transformers (**ViT**) and Multi-task Cascaded Convolutional Networks (**MTCNN**) to detect face-swap deepfakes (**DFDC**) and synthetic AI-generated faces (**StyleGAN / FFHQ**), complete with self-attention rollout heatmap visualizer.

🚀 **Live Web Application Deployment:** [https://deep-fake-detection-vivit1810.streamlit.app/](https://deep-fake-detection-vivit1810.streamlit.app/)

---

## 📌 1. Project Overview & Description

With the rapid progression of modern generative AI, distinguishing real human faces from manipulated or synthesized media has become a critical cybersecurity and media authenticity challenge. Standard deepfake detectors often fail when encountering new forgery methods or process full video frames containing irrelevant background data.

This project delivers a **dual-capability, explainable deepfake detection pipeline**:

1. **Facial Isolation & Context Expansion:** Utilizes **MTCNN** face detection with a **+30% margin expansion** around bounding boxes. This captures critical facial boundary seams, forehead blending lines, and jawline contours where manipulation artifacts concentrate.
2. **Vision Transformer (ViT) Backbone:** Leverages Hugging Face's `google/vit-base-patch16-224-in21k` architecture. Instead of relying purely on local pixel textures like traditional CNNs, ViT models model long-range spatial dependencies across grid patches (14x14 patches of 16x16 pixels).
3. **Dual-Capability Forgery Detection:** Trained via mixed transfer learning (`ConcatDataset`) to detect both:
   * **Face-Swap Deepfakes** (e.g., DeepFake Detection Challenge / DFDC).
   * **Fully AI-Synthesized Faces** (e.g., StyleGAN / FFHQ synthetic profiles).
4. **Explainable AI (XAI) via Attention Rollout:** Implements the **Self-Attention Rollout algorithm** across all 12 transformer layers and heads. It projects the model's focus back onto the input face, rendering interactive thermal heatmaps:
   * 🔴 **Red Hotspots:** High-influence regions (blending seams, unnatural skin transitions, facial overlays).
   * 🔵 **Blue Cool Zones:** Low-influence regions ignored during decision making.
5. **Multi-Media Streamlit Interface:** Operates on both static image scans (`.jpg`, `.png`) and temporal video files (`.mp4`, `.mov`, `.avi`) with dynamic frame-by-frame inspection.

---

## 📁 2. Repository File Structure & Component Directory

Below is the complete list of files and directories in this repository, along with their purpose:

```text
DeepFake_Detection_ViViT/
├── app.py                      # Production Streamlit application & inference engine
├── dfd_ViT.ipynb               # Master Jupyter Notebook (End-to-end pipeline, training, & evaluation)
├── requirements.txt            # Python package dependencies
├── LICENSE                     # Project license (MIT License)
├── .gitattributes              # Git Large File Storage (LFS) rules for .pth model weights
├── .gitignore                  # Git untracked file exclusions
├── .lfsconfig                  # Git LFS repository configuration
├── Model/                      # Directory storing trained model checkpoints
│   ├── best_vit_deepfake.pth   # Model 1 (v1 baseline: Trained on initial DFDC subset)
│   ├── best_vit_deepfake_2.pth # Model 2 (v2 baseline: Dedicated DFDC face-swap model)
│   └── best_vit_deepfake_3.pth # Model 3 (v3 production: Mixed fine-tuned dual-capability model)
├── documentation/              # Technical documentation & project tracking
│   ├── DeepFake_Report.md      # Full architecture report, methodology, & benchmarks
│   └── progression.md          # Phase-by-phase task tracking log
├── dataset/                    # Storage root for processed training & validation datasets
└── dataset_archives/           # Raw dataset compressed archives
```

### Detailed File Explanations

* **[`app.py`]:** The primary production Streamlit frontend script. It loads `Model/best_vit_deepfake_3.pth`, initializes CPU-isolated MTCNN face detection, handles image upload and video sampling workflows, computes attention rollout heatmaps, and renders custom-styled metric cards.
* **[`dfd_ViT.ipynb`]:** The comprehensive master notebook containing all ML pipeline phases: metadata parsing, face cropping, PyTorch dataset loaders, ViT model wrapping, AdamW optimization with cosine warmup, gradient accumulation (4 steps), mixed dataset fine-tuning, cross-dataset evaluation, and side-by-side performance comparisons.
* **[`requirements.txt`]:** Lists all required Python packages (`torch`, `torchvision`, `transformers`, `facenet-pytorch`, `streamlit`, `opencv-python`, `scikit-learn`, `pillow`, `pandas`, `tqdm`).
* **[`Model/`]:** Contains all three generated `.pth` PyTorch model checkpoints (~345.6 MB each).
  * **[`Model/best_vit_deepfake.pth`]:** Model 1 (v1 baseline trained on early DFDC data).
  * **[`Model/best_vit_deepfake_2.pth`]:** Model 2 (v2 model trained on full DFDC dataset).
  * **[`Model/best_vit_deepfake_3.pth`]:** Model 3 (v3 production dual-capability model trained on DFDC + StyleGAN).
* **[`documentation/DeepFake_Report.md`]):** Formal project report outlining problem formulation, dataset reorganization, transfer learning strategy, catastrophic forgetting prevention, and experimental conclusions.
* **[`documentation/progression.md`](file:///Users/jellyfish/DeveloperLocal/DeepFake_Detection_ViViT/documentation/progression.md):** 30-step task tracking document recording development milestones from Phase 1 (Data Prep) through Phase 6 (StyleGAN Extension).
* **[`.gitattributes`](file:///Users/jellyfish/DeveloperLocal/DeepFake_Detection_ViViT/.gitattributes):** Tracks binary checkpoint `.pth` files via Git LFS so heavy model weights can be version-controlled without bloating standard Git commits.
* **[`LICENSE`](file:///Users/jellyfish/DeveloperLocal/DeepFake_Detection_ViViT/LICENSE):** Governs open-source usage under the MIT License terms.

---

## 🛠️ 3. Complete Process & Usage Guide

Follow these step-by-step instructions to set up, run, or train the project locally.

### Prerequisites
* **Python 3.9** or higher installed.
* **Git** & **Git LFS** installed.
* Hardware: Recommended NVIDIA GPU (CUDA) or Apple Silicon (M1/M2/M3/M4 via `mps` backend). CPU inference is supported as a fallback.

---

### Step 1: Clone the Repository & Fetch LFS Weights

```bash
git clone https://github.com/AbhoyShah1810/DeepFake_Detection_ViViT.git
cd DeepFake_Detection_ViViT

# Ensure Git LFS pulls the large model checkpoint files (.pth)
git lfs install
git lfs pull
```

---

### Step 2: Create a Virtual Environment & Install Dependencies

```bash
# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate.bat

# Upgrade pip and install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Run the Streamlit Web Application

Launch the local web app server:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

#### Using the App:
1. **Image Mode:** Upload any facial image (`.jpg`, `.jpeg`, `.png`). The app will:
   * Extract the primary face using MTCNN with margin expansion.
   * Run ViT inference and calculate authenticity probability.
   * Display a verdict card (REAL / FAKE with confidence percentage).
   * Render the Self-Attention Rollout heatmap overlay for explainability.
2. **Video Mode:** Upload a video asset (`.mp4`, `.avi`, `.mov`). The app will:
   * Sample 1 frame per second across the video timeline.
   * Extract facial crops and run batched ViT inference.
   * Compute sequence-averaged probability to render a final video verdict.
   * Provide an interactive frame slider to inspect individual attention maps frame-by-frame.

---

### Step 4: Re-training or Retesting via Jupyter Notebook

To inspect, run, or modify the model training pipeline:

```bash
jupyter notebook dfd_ViT.ipynb
```

The notebook guides you through:
* **Data Preprocessing:** MTCNN face extraction and margin calculation.
* **Model Training:** Initial fine-tuning on DFDC dataset (`Model/best_vit_deepfake_2.pth`).
* **Mixed Fine-Tuning:** Concatenating datasets via `ConcatDataset` and running warm-up head training followed by full unfreezing (`Model/best_vit_deepfake_3.pth`).
* **Cross-Evaluation:** Generating metrics tables across both validation/test splits.

---

## 📊 4. Model Comparison & Performance Statistics

During development, three model iterations were trained and evaluated across two distinct benchmark datasets:
1. **DFDC Validation Set** (Face-swap deepfakes: surgical facial overlay boundaries).
2. **StyleGAN Test Set** (20,000 images: FFHQ real faces vs. StyleGAN AI-synthesized faces).

### Overview of Model Differences

* **Model 1 (`best_vit_deepfake.pth` - v1 Baseline):**
  * Fine-tuned on an initial subset of the DFDC dataset.
  * Achieved high validation scores on early splits but exhibited a train-validation gap (0.1844), indicating mild overfitting.
* **Model 2 (`best_vit_deepfake_2.pth` - v2 DFDC Specialist):**
  * Fine-tuned exclusively on the full DFDC dataset using Gradient Accumulation (4 steps, effective batch size 32) and Cosine Warmup scheduling.
  * Achieved exceptional performance on face-swaps (**91.39% accuracy, 0.9738 ROC-AUC**).
  * **Limitation:** Failed completely on synthetic AI faces (**51.92% accuracy / 0.4958 ROC-AUC on StyleGAN**, equivalent to a random guess), demonstrating domain specificity.
* **Model 3 (`best_vit_deepfake_3.pth` - v3 Final Production Model):**
  * Fine-tuned using transfer learning from v2 on a concatenated dataset (`ConcatDataset`) combining DFDC and StyleGAN data.
  * **Catastrophic Forgetting Prevented:** Retained 100.4% of its face-swap performance on DFDC (**91.75% accuracy, 0.9752 ROC-AUC**).
  * **Superb Synthetic Face Detection:** Boosted StyleGAN detection from 51.92% to **99.25% accuracy (0.9995 ROC-AUC)**.
  * **Selected for Production:** Powering the live Streamlit web application.

---

### 📈 Complete Statistical Performance Table

| Model Checkpoint | Training Dataset / Methodology | Dataset Benchmark | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Val Loss | Status / Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1**<br>`best_vit_deepfake.pth` | Initial DFDC subset fine-tuning | **DFDC Val** | 92.33% | 92.30% | 92.66% | 92.48% | 0.9757 | 0.2440 | ⚠️ Early baseline (Mild overfitting) |
| **Model 2**<br>`best_vit_deepfake_2.pth` | Full DFDC dataset fine-tuning | **DFDC Val**<br>**StyleGAN Test** | 91.39%<br>51.92% | 91.44%<br>51.14% | 91.66%<br>86.43% | 91.55%<br>64.26% | 0.9738<br>0.4958 | 0.2199<br>— | 🎯 Face-Swap Specialist<br>❌ Fails on GAN faces |
| **Model 3**<br>`best_vit_deepfake_3.pth`<br>*(Production Model)* | Mixed fine-tuning (`ConcatDataset`: DFDC + StyleGAN) | **DFDC Val**<br>**StyleGAN Val**<br>**StyleGAN Test** | **91.75%**<br>**99.20%**<br>**99.25%** | **91.82%**<br>**99.30%**<br>**99.38%** | **91.97%**<br>**99.10%**<br>**99.11%** | **91.90%**<br>**99.20%**<br>**99.24%** | **0.9752**<br>**0.9997**<br>**0.9995** | 0.2693<br>0.0239<br>0.0245 | 🏆 **Production Selected**<br>✅ Dual-Capability (Swaps & GANs) |

---

### Key Takeaways from Model Benchmarks

1. **Zero Catastrophic Forgetting:** Training on mixed dataset batches rather than sequential task learning allowed Model 3 to preserve DFDC face-swap detection while acquiring synthetic face detection capability.
2. **Feature Synergies:** Incorporating FFHQ real human facial samples alongside DFDC real frames marginally enhanced DFDC validation accuracy (+0.35%) and ROC-AUC (+0.0014).
3. **Massive StyleGAN Gain:** Model 3 achieved a **+47.33% accuracy leap** on StyleGAN test sets compared to Model 2, establishing robust generalization across both face manipulation families.

---

## 📜 License

This project is open-source software licensed under the [MIT License](file:///Users/jellyfish/DeveloperLocal/DeepFake_Detection_ViViT/LICENSE).
