import os
import json
import streamlit as st
from pipeline import generate_optimized_ini
from utils import safe_json_loads

# ================================
# Wrapper pour Streamlit
# ================================
def traiter_prompt(project_dir: str, prompt: str, part: str) -> str:
    try:
        user_input = safe_json_loads(prompt)
    except json.JSONDecodeError as e:
        return f"Erreur de format JSON dans le prompt : {e}"
    
    return generate_optimized_ini(project_dir, user_input, part)

# ================================
# Interface Streamlit
# ================================
st.title("Web App from Notebook")

project_dir = st.text_input("Project repository :", value=os.getcwd())
problematic = st.text_area("Enter your problematic :", height=50)
model = st.text_area("Model (e.g., UnetVanilla, UnetSegmentor, DINOv2):", height=40)
optimizer = st.text_area("Optimizer (e.g., Adam, SGD):", height=50)
scheduler = st.text_area("Scheduler (e.g., StepLR, ReduceLROnPlateau):", height=50)
loss = st.text_area("Loss (e.g., CrossEntropyLoss, BCEWithLogitsLoss):", height=50)
training = st.text_area("Training (e.g., Jaccard, F1-score):", height=50)
data = st.text_area("Data (e.g., binary masks, multi-class):", height=50)

if st.button("Generate INI"):
    prompt = json.dumps({
        "Problematic": problematic,
        "Model": model,
        "Optimizer": optimizer,
        "Scheduler": scheduler,
        "Loss": loss,
        "Training": training,
        "Data": data
    }, indent=4)
    resultat = traiter_prompt(project_dir, prompt, "Model")
    st.text_area("Result :", value=resultat, height=500)
