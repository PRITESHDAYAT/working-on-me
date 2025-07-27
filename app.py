import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

segments = [
"Modern", "Traditional", "Classic", "Ultramodern", "Creative Classic", "Fusion", "Arabic",
"International", "Couture", "Chain Store", "Independent Store", "Online Store",
"Franchise Store", "Wholesale Distributor", "Manufacturer / Export House", "Generic Fancy",
"Classic Illusion", "Composite Diamond", "LGD Special Cut", "LGD Classic", "Limited Edition",
"Convertible Jewellery", "Statement Piece", "Minimalist", "Cocktail Ring", "Bridal Collection",
"Everyday Wear", "Festive Jewellery", "Office Wear", "Religious Jewellery", "Kids Jewellery",
"Men’s Jewellery", "Gifting Collection", "Art Deco", "Victorian", "Antique Finish", "Filigree",
"Enamel Work", "Temple Jewellery", "Kundan", "Meenakari", "Jadau", "Granulation", "Hand Engraving",
"Laser Cutting", "3D CAD Design", "Rhodium Plating", "Polishing & Finishing"
]

st.title("Jewelry Segment Classifier")
uploaded_file = st.file_uploader("Upload a Jewelry Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(text=segments, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).squeeze()

    top_probs, top_labels = torch.topk(probs, k=5)
    st.subheader("Top Matching Segments:")
    for idx in range(top_labels.size(0)):
        st.write(f"{segments[top_labels[idx]]} ({top_probs[idx]*100:.2f}%)")
