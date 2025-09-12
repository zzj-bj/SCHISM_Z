import os
import json
import streamlit as st
from pipeline import generate_optimized_ini
from utils_gen import safe_json_loads

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

st.title("Generate your SCHISM INI file with AI")

project_dir = st.text_input("Project repository :", value=os.getcwd())
API_key = st.text_input("API Key :", type="password")
problematic = st.text_area("Enter your problematic :", height=50, value = "1 GPU with 12GB VRAM, and I want the best performance possible for my config. I need something fast and quick")
model = st.text_area("Model (e.g., UnetVanilla, UnetSegmentor, DINOv2):", height=40, value ="I have heard of UnetVanilla. I also want to test with 8 n_block")
optimizer = st.text_area("Optimizer (e.g., Adam, SGD):", height=50, value = "RMSprop")
scheduler = st.text_area("Scheduler (e.g., StepLR, ReduceLROnPlateau):", height=50, value = "anything")
loss = st.text_area("Loss (e.g., CrossEntropyLoss, BCEWithLogitsLoss):", height=50, value = "I don't know")
training = st.text_area("Training (e.g., Jaccard, F1-score):", height=50, value = "for metrics: all. i want it to run 40 epochs")
data = st.text_area("Data (e.g., binary masks, multi-class):", height=50, value = "I have 1200 images, grayscale i guess. i have several classes")

st.session_state['API_key'] = API_key
st.session_state['NbTokens'] = 0

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
    st.text_area("Result :", value=resultat, height=1000)
