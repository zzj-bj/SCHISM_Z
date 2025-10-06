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
st.markdown(
    """
    <style>
    .main {
        max-width: 1400px;
        margin: auto;
        display: flex;
        flex-direction: row;
        align-items: stretch;
    }
    .left-col, .right-col, .center-col {
        padding: 10px;
    }
    .center-col {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# === Title on two lines ===
st.markdown("<h1 style='text-align: center;'>GARVIS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Generative Assistant for Reliable Variables and INI Setup</h2>", unsafe_allow_html=True)

col_inputs, col_buttons, col_results = st.columns([2.5, 1, 2.5])
with col_inputs:
    project_dir = st.text_input("Project repository :", value=os.getcwd())
    API_key = st.text_input("API Key :", type="password")
    problematic = st.text_area("Enter your problematic :", height=50, width=500, value = "1 GPU with 12GB VRAM, and I want the best performance possible for my config. I need something fast and quick")
    model = st.text_area("Model (e.g., UnetVanilla, UnetSegmentor, DINOv2):", height=40, value ="I have heard of UnetVanilla. I also want to test with 8 n_block")
    optimizer = st.text_area("Optimizer (e.g., Adam, SGD):", height=50, value = "RMSprop")
    scheduler = st.text_area("Scheduler (e.g., StepLR, ReduceLROnPlateau):", height=50, value = "anything")
    loss = st.text_area("Loss (e.g., CrossEntropyLoss, BCEWithLogitsLoss):", height=50, value = "I don't know")
    training = st.text_area("Training (e.g., Jaccard, F1-score):", height=50, value = "for metrics: all. i want it to run 40 epochs")
    data = st.text_area("Data (e.g., binary masks, multi-class):", height=50, value = "I have 1200 images, grayscale i guess. i have several classes")

st.session_state['API_key'] = API_key
st.session_state['NbTokens'] = 0

with col_buttons:
    st.markdown("<div style='display:flex; align-items:center; justify-content:center; height:100%;'>", unsafe_allow_html=True)
    generate_button = st.button("Generate INI", key="generate_ini_btn")
    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    result_placeholder = st.empty()

if generate_button:
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
    result_placeholder.text_area("Result :", value=resultat, height=1000)
