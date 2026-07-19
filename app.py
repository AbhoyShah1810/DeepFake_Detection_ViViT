import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
import pandas as pd
from facenet_pytorch import MTCNN
from transformers import ViTImageProcessor, ViTModel
from torchvision import transforms
import tempfile

# Set page configuration to centered layout and set title
st.set_page_config(page_title="DeepFake Detector", layout="centered", initial_sidebar_state="collapsed")

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;600&display=swap');
    
    /* Overall Page Background & Styling - Soothing Warm Cream */
    .stApp {
        background-color: #F6EFEA !important;
        color: #3C3633 !important;
    }
    
    /* Header Area */
    .title-text {
        font-family: 'Outfit', sans-serif;
        color: #3C3633;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
        text-align: center;
    }
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #746964;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        opacity: 0.9;
        text-align: center;
    }
    
    /* Premium Soothing Muted Cards */
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #EAE0DA !important;
        border-radius: 28px;
        padding: 1.8rem;
        box-shadow: 0 10px 45px rgba(60, 54, 51, 0.03) !important;
        margin-bottom: 1.5rem;
        max-width: 450px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .fake-accent {
        border-top: 6px solid #C62828 !important;  /* Crimson Red */
    }
    
    .real-accent {
        border-top: 6px solid #1E6B27 !important;  /* Forest Green */
    }
    
    .card-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #746964;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin-top: 0.4rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .color-fake {
        color: #C62828 !important;
    }
    
    .color-real {
        color: #1E6B27 !important;
    }
    
    /* Footer Styling */
    .app-footer {
        text-align: center;
        margin-top: 5rem;
        padding: 2.5rem 0;
        border-top: 1px solid #EAE0DA;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #746964;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* File Uploader Container */
    .stFileUploader {
        background-color: #FFFFFF !important;
        border: 2px dashed #D3C4B9 !important;
        border-radius: 28px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 30px rgba(60, 54, 51, 0.01) !important;
    }
    
    /* Pill buttons for Streamlit */
    div.stButton > button {
        background-color: #F39F5A !important; /* Soft Peach */
        color: #3C3633 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border: 1px solid #E0D4CC !important;
        border-radius: 30px !important;
        padding: 0.6rem 2.5rem !important;
        box-shadow: 0 4px 14px rgba(243, 159, 90, 0.15) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        max-width: 400px !important;
        margin: 1.5rem auto 0 auto !important;
        display: block !important;
    }
    div.stButton > button:hover {
        background-color: #E8BCB9 !important; /* Dusty Pink */
        color: #3C3633 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(232, 188, 185, 0.25) !important;
        border-color: #D3C4B9 !important;
    }
</style>


""", unsafe_allow_html=True)


# ----------------- MODEL DEFINITION -----------------

class ViTDeepfakeClassifier(nn.Module):
    def __init__(self, vit_model, dropout_rate=0.2):
        super().__init__()
        self.vit = vit_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.vit.config.hidden_size, 1)
        
    def forward(self, pixel_values):
        # FIX: Ensure output_attentions=True passes forward during inference rollouts
        outputs = self.vit(pixel_values=pixel_values, output_attentions=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits, outputs.attentions # Return weights alongside logits

# ----------------- CACHED RESOURCES -----------------

@st.cache_resource
def load_models():
    model_name = "google/vit-base-patch16-224-in21k"
    
    # Establish PyTorch Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Load pre-trained HF components
    processor = ViTImageProcessor.from_pretrained(model_name)
    vit_base = ViTModel.from_pretrained(model_name, attn_implementation="eager")
    
    # Create customized classifier architecture
    model = ViTDeepfakeClassifier(vit_base, dropout_rate=0.2)
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################

    # Load our checkpoint weights
    checkpoint_path = "Model/best_vit_deepfake_3.pth"
    weights_loaded = False
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        weights_loaded = True
        
    model.to(device)
    model.eval()
    #################################################################################################################
    #################################################################################################################
    #################################################################################################################
    
    # Initialize face detector on CPU (workaround for MPS pooling errors)
    detector = MTCNN(keep_all=True, device='cpu')
    
    return model, processor, detector, device, weights_loaded

# Load components
model, processor, detector, device, weights_loaded = load_models()



# ----------------- HEADER AREA -----------------

st.markdown('<div class="title-text">Vision Transformer Deepfake Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Dual-Capability Detector: Detecting Face-Swap Manipulations (DFDC) & Synthetic GAN Faces (StyleGAN) using explainable Self-Attention rollout mapping.</div>', unsafe_allow_html=True)

# ----------------- CORE LOGIC FUNCTIONS -----------------

def get_largest_face(image, margin=0.30):
    """
    Detect faces using MTCNN, isolate the largest face by bounding box area,
    apply box expansion logic (+30% margin), and return the cropped face.
    """
    boxes, probs = detector.detect(image)
    if boxes is None or len(boxes) == 0:
        return None
    
    # Calculate bounding box areas
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    largest_idx = np.argmax(areas)
    box = boxes[largest_idx]
    
    # Expand box coordinates with margin
    img_w, img_h = image.size
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    x1_new = max(0, int(x1 - w * (margin / 2)))
    y1_new = max(0, int(y1 - h * (margin / 2)))
    x2_new = min(img_w, int(x2 + w * (margin / 2)))
    y2_new = min(img_h, int(y2 + h * (margin / 2)))
    
    cropped_face = image.crop((x1_new, y1_new, x2_new, y2_new))
    return cropped_face

def compute_attention_rollout(pixel_values):
    """
    Implement Attention Rollout algorithm across all 12 layers of the Vision Transformer
    to generate an explainable heatmap projected onto original inputs.
    """
    with torch.no_grad():
        # Extracted from customized forward format configuration
        logits, attentions = model(pixel_values)
        
    seq_len = attentions[0].shape[-1]
    rollout = np.eye(seq_len)
    
    for layer_attn in attentions:
        # Average attention maps across all 12 heads
        attn_heads = layer_attn[0].cpu().numpy()  # Shape: (12, 197, 197)
        attn_avg = np.mean(attn_heads, axis=0)
        
        # Introduce residual identity: A_res = 0.5 * I + 0.5 * A_avg
        I = np.eye(seq_len)
        a_res = 0.5 * attn_avg + 0.5 * I
        
        # Normalize rows so row vectors sum to 1
        a_res = a_res / np.sum(a_res, axis=-1, keepdims=True)
        
        # Multiply across layers to get combined matrix
        rollout = np.matmul(a_res, rollout)
        
    # Extract rollout attention from the CLS token (index 0) to all patch tokens (index 1 to 196)
    cls_attn = rollout[0, 1:]
    
    # Reshape back to 14x14 grid
    grid_size = int(np.sqrt(len(cls_attn)))
    heatmap = cls_attn.reshape(grid_size, grid_size)
    
    # Normalize values between 0 and 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, logits

def overlay_heatmap(heatmap, face_image):
    """
    Superimpose the attention rollout heatmap onto the cropped face image.
    """
    face_np = np.array(face_image)
    h, w, c = face_np.shape
    
    # Resize heatmap to match crop size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to uint8 range (0-255) and apply JET color mapping
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap onto the original image (50% opacity blend)
    blended_img = cv2.addWeighted(face_np, 0.5, heatmap_color, 0.5, 0)
    return Image.fromarray(blended_img)

def render_verdict_card(prob):
    """
    Renders premium glassmorphic layout metrics depending on target probability weights.
    """
    is_fake = prob < 0.5
    confidence = (1.0 - prob) if is_fake else prob
    
    accent_class = "fake-accent" if is_fake else "real-accent"
    label_text = "FAKE VERDICT DETECTED" if is_fake else "REAL VERDICT CONFIRMED"
    color_class = "color-fake" if is_fake else "color-real"
    
    st.markdown(f"""
    <div class="metric-card {accent_class}">
        <div class="card-label">{label_text}</div>
        <div class="card-value {color_class}">{confidence*100:.2f}% Confidence</div>
    </div>
    """, unsafe_allow_html=True)

def render_heatmap_explanation():
    """
    Renders a clean card explaining how the attention heatmap highlights classification influences.
    """
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #EAE0DA; border-radius: 20px; padding: 1.5rem; margin-top: 1.5rem; box-shadow: 0 4px 20px rgba(60, 54, 51, 0.02);">
        <h4 style="font-family: 'Outfit', sans-serif; color: #3C3633; margin-top: 0; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
            💡 Understanding the Attention Heatmap
        </h4>
        <p style="font-family: 'Inter', sans-serif; font-size: 0.95rem; color: #5C524D; line-height: 1.5; margin-bottom: 0.8rem;">
            The Vision Transformer (ViT) processes face structures in grid-like patches. The attention map rolls out how much focus the model placed on each region when confirming its verdict:
        </p>
        <ul style="font-family: 'Inter', sans-serif; font-size: 0.92rem; color: #5C524D; line-height: 1.5; margin-bottom: 0; padding-left: 1.2rem;">
            <li style="margin-bottom: 0.4rem;">
                <strong style="color: #C62828;">Red & Warm Hotspots:</strong> Indicate regions of <strong>high influence</strong>. These are the facial features, contours, or structural anomalies (like blending lines around the eyes, jaw, or nose) that the model focused on to make its decision.
            </li>
            <li style="margin-bottom: 0.4rem;">
                <strong style="color: #1E6B27;">Blue & Cool Regions:</strong> Represent areas of <strong>low influence</strong>. These portions of the image were largely ignored by the classification head.
            </li>
            <li>
                <strong>Spotting Deepfakes:</strong> Genuine videos tend to show uniform attention concentrated on key natural facial details (eyes, nose, mouth movements). Deepfakes often generate localized high-attention spikes around artificial blending seams, skin-color mismatches, and boundary contours where facial overlays were spliced.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PIPELINE SELECTION & ROUTING -----------------
# Universal media uploader accepting images or video assets
uploaded_file = st.file_uploader("Upload Image or Video Blueprint...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Render the custom Upload Card theme on the main page when no file is uploaded:
if uploaded_file is None:
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2.5rem; background: #FFFFFF; border-radius: 32px; border: 1px solid #EAE0DA; box-shadow: 0 10px 45px rgba(60, 54, 51, 0.03); max-width: 420px; margin: 2rem auto 1rem auto; text-align: center;">
        <svg width="120" height="90" viewBox="0 0 120 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 1.5rem;">
            <!-- Orange folder body -->
            <path d="M10 25C10 20.5817 13.5817 17 18 17H45L55 27H102C106.418 27 110 30.5817 110 35V82C110 86.4183 106.418 90 102 90H18C13.5817 90 10 86.4183 10 82V25Z" fill="#FFAA88" stroke="#3C3633" stroke-width="3" stroke-linejoin="round"/>
            <path d="M18 27H102" stroke="#3C3633" stroke-width="3"/>
            <path d="M22 35C22 33.8954 22.8954 33 24 33H40C41.1046 33 42 33.8954 42 35V45H22V35Z" fill="white" opacity="0.4"/>
            <!-- Arrow upload indicator -->
            <path d="M60 70V50M60 50L53 57M60 50L67 57" stroke="#3C3633" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 600; color: #3C3633; margin-bottom: 0.4rem;">Upload from device</div>
        <div style="font-size: 0.85rem; color: #746964; line-height: 1.4;">Drag and drop file here<br>Max file size up to 20Mb</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    # --- PIPELINE A: IMAGE FILE PROCESSING ---
    if file_type == "image":
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image Blueprint", use_container_width=True)
            
        cache_key = f"image_{uploaded_file.name}"
        if cache_key not in st.session_state:
            with col2:
                with st.spinner("Extracting region of interest (Face Isolation)..."):
                    face = get_largest_face(image)
                    
                if face is not None:
                    inputs = processor(images=face, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].to(device)
                    
                    with st.spinner("Analyzing artifacts & running self-attention mechanics..."):
                        heatmap, logits = compute_attention_rollout(pixel_values)
                        
                    vis_image = overlay_heatmap(heatmap, face)
                    prob = torch.sigmoid(logits).item()
                    st.session_state[cache_key] = {
                        "face": face,
                        "vis_image": vis_image,
                        "prob": prob
                    }
                else:
                    st.session_state[cache_key] = None
                    
        cached_data = st.session_state.get(cache_key)
        if cached_data is not None:
            face = cached_data["face"]
            vis_image = cached_data["vis_image"]
            prob = cached_data["prob"]
            
            with col2:
                render_verdict_card(prob)
                
            st.markdown("---")
            st.markdown("<h3 style='text-align: center; margin-top: 1.5rem; margin-bottom: 1.5rem;'>Self-Attention Map Interpretability Rollout</h3>", unsafe_allow_html=True)
            
            is_fake = prob < 0.5
            confidence = (1.0 - prob) if is_fake else prob
            verdict_str = "FAKE" if is_fake else "REAL"
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    face, 
                    caption=f"Isolated Face Profile — Unique Judgement: {verdict_str} ({confidence*100:.2f}% Confidence)", 
                    use_container_width=True
                )
            with c2:
                st.image(
                    vis_image, 
                    caption="Attention Hotspot Overlay", 
                    use_container_width=True
                )
            render_heatmap_explanation()
        else:
            with col2:
                st.error("No faces identified in the blueprint. Please provide a clear profile scan.")
                
    # --- PIPELINE B: VIDEO FILE PROCESSING ---
    elif file_type == "video":
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_file)
            
        cache_key = f"video_{uploaded_file.name}"
        if cache_key not in st.session_state:
            with col2:
                st.markdown("### Temporal Inspection Progress")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                attention_slot = st.empty()
                
                # Save temporary video file safely
                suffix = '.' + uploaded_file.name.split('.')[-1]
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tfile.write(uploaded_file.read())
                tfile.close()
                
                # Read video file via OpenCV container
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sample exactly 1 frame per second to speed up evaluation loops
                sample_rate = max(1, int(fps)) if fps > 0 else 30
                frame_idx = 0
                processed_probabilities = []
                processed_frames_data = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check sample timing index alignment
                    if frame_idx % sample_rate == 0:
                        # Update progress metric
                        current_pct = min(1.0, frame_idx / max(1, total_frames))
                        progress_bar.progress(current_pct)
                        status_text.text(f"Processing structural integrity matrix... [Frame {frame_idx}/{total_frames}]")
                        
                        # Convert OpenCV Frame layout (BGR) to PIL Image layout (RGB)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        
                        face = get_largest_face(pil_img)
                        if face is not None:
                            inputs = processor(images=face, return_tensors="pt")
                            pixel_values = inputs['pixel_values'].to(device)
                            
                            heatmap, logits = compute_attention_rollout(pixel_values)
                            prob = torch.sigmoid(logits).item()
                            processed_probabilities.append(prob)
                            
                            # Generate attention visualization snapshot
                            vis_image = overlay_heatmap(heatmap, face)
                            attention_slot.image(vis_image, caption="Live Sequence Attention Heatmap", use_container_width=True)
                            
                            # Save frame details
                            processed_frames_data.append({
                                "face_crop": face,
                                "heatmap_vis": vis_image,
                                "prob": prob
                            })
                    
                    frame_idx += 1
                    
                cap.release()
                
                # Cleanup temporary file
                try:
                    os.unlink(tfile.name)
                except Exception as e:
                    pass
                    
                progress_bar.empty()
                status_text.empty()
                attention_slot.empty()
                
                if len(processed_probabilities) > 0:
                    st.session_state[cache_key] = {
                        "probabilities": processed_probabilities,
                        "frames_data": processed_frames_data
                    }
                else:
                    st.session_state[cache_key] = None
                    
        cached_data = st.session_state.get(cache_key)
        if cached_data is not None:
            processed_probabilities = cached_data["probabilities"]
            processed_frames_data = cached_data["frames_data"]
            mean_probability = np.mean(processed_probabilities)
            
            with col2:
                st.markdown("### Final Sequence Analysis")
                render_verdict_card(mean_probability)
                
            st.markdown("---")
            st.markdown("<h3 style='text-align: center; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Key Frame Interpretability Analysis</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #746964; margin-bottom: 1.5rem;'>Use the slider below to step through all analyzed frames and inspect individual model decisions.</p>", unsafe_allow_html=True)
            
            # Slider to select frame
            num_frames = len(processed_frames_data)
            selected_frame_number = st.slider(
                "Select Frame to Inspect",
                min_value=1,
                max_value=num_frames,
                value=1,
                step=1,
                format="Frame %d"
            )
            
            # Get selected frame data
            frame_data = processed_frames_data[selected_frame_number - 1]
            frame_prob = frame_data["prob"]
            is_frame_fake = frame_prob < 0.5
            frame_confidence = (1.0 - frame_prob) if is_frame_fake else frame_prob
            frame_verdict = "FAKE" if is_frame_fake else "REAL"
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    frame_data["face_crop"],
                    caption=f"Face Profile (Frame {selected_frame_number}) — Unique Judgement: {frame_verdict} ({frame_confidence*100:.2f}% Confidence)",
                    use_container_width=True
                )
            with c2:
                st.image(
                    frame_data["heatmap_vis"],
                    caption=f"Attention Hotspots (Frame {selected_frame_number})",
                    use_container_width=True
                )
            render_heatmap_explanation()
        else:
            with col2:
                st.error("No valid facial profile structures could be isolated anywhere in this video asset.")

# ----------------- FOOTER -----------------
st.markdown("""
<div class="app-footer">
    © 2026 jf.dev, built on caffeine + infinite curiosity to know the unknown.
</div>
""", unsafe_allow_html=True)