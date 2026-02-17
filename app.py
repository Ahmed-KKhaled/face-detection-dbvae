# app.py
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# -----------------------------
# Load Model
# -----------------------------
from Debiasing_Facial_Detection_Systems import DB_VAE, ConvBlock, FaceDecoder

torch.serialization.add_safe_globals([DB_VAE])  # declare class as safe
dbvae = torch.load("dbvae_model.pth", map_location="cpu", weights_only=False)
dbvae.eval()



# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Face Detection DB-VAE Demo")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Updated image display (no deprecated warning)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # same size used in training
        transforms.ToTensor(),         # converts to [C,H,W] and scales 0-1
    ])

    img_tensor = transform(img).unsqueeze(0)  # [1,3,64,64]

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.inference_mode():
        logit = dbvae.predict(img_tensor)
        prob = torch.sigmoid(logit).item()

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("Prediction Result")

    if prob > 0.5:
        st.success(f"Face Detected ✅  (Confidence: {prob:.2f})")
    else:
        st.error(f"Not a Face ❌  (Confidence: {prob:.2f})")
