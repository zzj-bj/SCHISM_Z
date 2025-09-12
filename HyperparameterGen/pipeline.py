import re
import requests
import streamlit as st
from utils_gen import estimate_call_cost
from collections import defaultdict
from processing import (
    read_ini_files,
    determine_component_choice_model,
    determine_component_choice_loss,
    fetch_and_parse_docs,
    determine_component_choice_O_S_L,
    determine_component_choice_training,
    determine_data_parameters
)

#############################################
# MAIN FUNCTION TO GENERATE THE INI
#############################################
def generate_optimized_ini(root_folder, user_input, part) -> str:
    """
    Generates an optimized INI file by:
      1. Building a knowledge base from local INI files.
      2. Fetching and parsing GitHub documentation to extract available options.
      3. Using the provided dictionary for user input.
      4. For each component (optimizer, scheduler, loss), querying the LLM to determine the selected option.
      5. For each selected component, fetching external documentation and generating an INI snippet.
      6. Combining the snippets with generic sections for [Model], [Training], [Data] to produce the final INI file.
    """
    # 1. Build knowledge base and fetch documentation.
    #knowledge_base = read_ini_files(root_folder, part)
    #documentation = fetch_github_docs(github_urls, external_urls=None)

    def fetch_dynamic_available_opts():
        url_inimd = "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/ini.md"
        url_map = {
            "UnetSegmentor": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/UnetSegmentor.md",
            "UnetVanilla": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/UnetVanilla.md",
            "DINOv2": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/DINOv2.md"
        }
        sections = defaultdict(list)

        # === 1. OPTIONS GLOBALES (depuis ini.md) ===
        try:
            response = requests.get(url_inimd)
            response.raise_for_status()
            text = response.text

        # Models
            model_choices = re.search(r"Supported models:\s*((?:\s+- `[^`]+.*\n)+)", text)
            if model_choices:
                m_choices = re.findall(r"- `([^`]+)`", model_choices.group(1))
                sections['Model'] = m_choices

        # Optimizer
            optimizer_choices = re.search(r"## \[Optimizer\](.*?)##", text, re.DOTALL)
            if optimizer_choices:
                opts = re.findall(r"- `([^`]+)`", optimizer_choices.group(1))
                sections['Optimizer'] = opts

        # Scheduler
            scheduler_choices = re.search(r"## \[Scheduler\](.*?)##", text, re.DOTALL)
            if scheduler_choices:
                scheds = re.findall(r"- `([^`]+)`", scheduler_choices.group(1))
                sections['Scheduler'] = scheds

        # Loss
            loss_choices = re.search(r"## \[Loss\](.*?)(?=\n##|\Z)", text, re.DOTALL)
            if loss_choices:
                lines = loss_choices.group(1).strip().split("\n")
                losses = []
                for line in lines:
                    match = re.match(r"\s*-\s+`([^`]+)`", line)
                    if match:
                        losses.append(match.group(1))
                    elif losses:
                        break
                sections["Loss"] = losses

        # Training metrics
            training_choices = re.search(r"- `metrics`:([^#\n]+)", text)
            if training_choices:
                metrics = re.findall(r"`([^`]+)`", training_choices.group(1))
                sections["Training"] = metrics

        except Exception as e:
            print(f"[ERREUR] Impossible de charger ini.md : {e}")

            # === 2. PARAMÈTRES PAR MODÈLE (docs séparées) ===
        model_params = {}
        for model_name, url in url_map.items():
            try:
                response = requests.get(url)
                response.raise_for_status()
                text = response.text
                params_match = re.search(r"\| *Parameter *\|.*?\n(\|[-| ]+\|\n(?:\|.*\n)+)", text)
                if params_match:
                    table_text = params_match.group(1)
                    lines = table_text.strip().split('\n')

                    # Extraire uniquement les noms de paramètres de la première colonne
                    params = []
                    for line in lines:
                        match = re.match(r"\|\s*`([^`]+)`\s*\|", line)
                        if match:
                            params.append(match.group(1))

                    model_params[model_name] = params
            except Exception as e:
                print(f"[ERREUR] Impossible de charger {model_name} : {e}")
        return dict(sections), model_params
    
    st.session_state['NbTokens'] += estimate_call_cost(user_input, model="gpt-4o")
    print(st.session_state.get('NbTokens'))

    available_opts, model_docs = fetch_dynamic_available_opts()
    
    # Example usage:
    github_url = "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/ini.md"
    docs_sections = fetch_and_parse_docs(github_url)
    #print(f"[DEBUG] Sections détectées dans docs_sections : {list(docs_sections.keys())}")

    # 3. Determine component choices based on user input.
    #print("___")
    # (nom, fonction, liste d'arguments personnalisés à fournir)
    components = [
        ('Model', determine_component_choice_model, lambda part: (
            user_input,
            available_opts.get(part),
            model_docs,
            docs_sections.get(part),
            read_ini_files(root_folder, part),
            part
        )),
        ('Optimizer', determine_component_choice_O_S_L, lambda part: (
            user_input,
            available_opts.get(part),
            docs_sections.get(part),
            read_ini_files(root_folder, part),
            part
        )),
        ('Scheduler', determine_component_choice_O_S_L, lambda part: (
            user_input,
            available_opts.get(part),
            docs_sections.get(part),
            read_ini_files(root_folder, part),
            part
        )),
        ('Training', determine_component_choice_training, lambda part: (
            user_input,
            available_opts.get(part),
            docs_sections.get(part),
            read_ini_files(root_folder, part),
            part
        )),
        ('Loss', determine_component_choice_loss, lambda part: (
            user_input,
            available_opts.get(part),
            read_ini_files(root_folder, part),
            part
        )),
        ('Data', determine_data_parameters, lambda part: (
            user_input,
            part
        ))
        # ('Data Augmentation', determine_data_augmentation_section, lambda part: (
        #     user_input
        # ))
    ]
    print(st.session_state.get('NbTokens'))
    outputs = []
    # Exécution en boucle
    for part, func, arg_builder in components:
        result = func(*arg_builder(part))
        outputs.append(result)

    return "\n\n".join(outputs)
#############################################