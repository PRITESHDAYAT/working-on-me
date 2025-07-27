
import streamlit as st
from PIL import Image
import os
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Load model and processor
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# Load segments
with open("segments.txt", "r") as f:
    segments = [line.strip() for line in f.readlines()]

st.title("Jewelry Segment Classifier")
uploaded_file = st.file_uploader("Upload a Jewelry Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(text=segments, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze()

    # Show top 5 matches
    top_probs, top_labels = torch.topk(probs, k=5)
    st.subheader("Top Matching Segments:")
    for idx in range(top_labels.size(0)):
        st.write(f"{segments[top_labels[idx]]} ({top_probs[idx]*100:.2f}%)")
