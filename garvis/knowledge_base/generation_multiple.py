import configparser
import random
from pathlib import Path
import re
import requests
from collections import defaultdict
from datetime import datetime

#===========================================================================
# Define possible choice
model_types = ['UnetVanilla', 'UnetSegmentor', 'DINOv2']
optimizers = ['Adagrad', 'Adam', 'AdamW', 'NAdam', 'RMSprop', 'RAdam', 'SGD']
schedulers = ['StepLR', 'MultiStepLR', 'ExponentialLR', 'PolynomialLR',
              'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR',
              'CosineAnnealingWarmRestarts']
losses = ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'NLLLoss']
activations = ['relu', 'LeakyReLU', 'sigmoid', 'tanh']

# Dynamic parameter depending of the possible choice inside optimizer
optimizer_extra_args = {
    "RMSprop": {"lr": 0.01, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0,
                "momentum": 0.0, "centered": False, "capturable": False,
                "maximize": False, "differentiable": False},
    "Adam": {"lr": 0.001, "betas":(0.9, 0.999), "eps": 1e-8, "weight_decay": 0,
              "amsgrad": False, "maximize": False, "differentiable": False, "fused": False},
    "SGD": {"lr": 0.01, "momentum": 0.9, "dampening": 0,
            "weight_decay": 0.0005, "nesterov": False, "maximize": False,
            "differentiable": False},
    "Adagrad": {"lr": 0.01, "lr_decay": 0, "weight_decay": 0.0,
                "initial_accumulator_value": 0.0, "eps": 1e-10},
    "AdamW": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
              "weight_decay": 0.01, "amsgrad": False, "amsgrad": False},
    "NAdam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
              "weight_decay": 0.0, "momentum_decay": 0.004},
    "RAdam": {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
              "weight_decay": 0.0},
}

# Dynamic parameter depending of the possible choice inside scheduler
scheduler_extra_args = {
    "StepLR": {"step_size": 10, "gamma": 0.1},
    "MultiStepLR": {"milestones": "[30, 80]", "gamma": 0.1},
    "ExponentialLR": {"gamma": 0.9},
    "PolynomialLR": {"total_iters": 100, "power": 2.0},
    "CosineAnnealingLR": {"T_max": 50, "eta_min": 0},
    "ReduceLROnPlateau": {"mode": "min", "factor": 0.5, "patience": 5,
                          "threshold": 1e-4, "threshold_mode": "rel",
                          "cooldown": 2, "min_lr": 1e-6,
                          "eps": 1e-8, "verbose": True},
    "OneCycleLR": {"max_lr": 0.01, "total_steps": 100},
    "CosineAnnealingWarmRestarts": {"T_0": 10, "T_mult": 2},
}

# Function to force string value
def stringify_dict(d):
    return {k: ("" if v is None else str(v)) for k, v in d.items()}

def answer_yes_or_no(message):
    while True:
        reponse = input(f"[?] {message} (y/n) ? : ").strip().lower()
        if reponse in ['yes', 'y', 'oui', 'o']:
            return True
        elif reponse in ['no', 'non', 'n']:
            return False
        else:
            print(f"incorrect answer !!!")

def ask_int(message, default=None):
    """Demande un entier avec vérification"""
    while True:
        user_input = input(f"[?] {message} : ").strip()
        if not user_input and default is not None:
            return default
        try:
            return int(user_input)
        except ValueError:
            print("⚠️ Please enter a valid integer.")

#==============================================================================

# Get the directory for hyperparameters files
source_directory = input(f"[?] Enter the directory for hyperparameters files: ").strip()
nb_batch = ask_int("Enter the number of config file to create", default=5)

# Nom unique basé sur la date/heure
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_configs_dir = Path(source_directory) / f"Configs_{timestamp}"

final_configs_dir.mkdir(parents=True, exist_ok=False)
print(f"Folder created: {final_configs_dir}")

if final_configs_dir is None:
    raise RuntimeError("Failed to create or locate a valid Configs directory.")

#==============================================================================

def choose_loss(num_classes):
    if num_classes <= 2:
        return "BCEWithLogitsLoss"
    else:
        return random.choice(["CrossEntropyLoss", "NLLLoss"])

def choose_scheduler(optimizer, epochs):
    if optimizer in ["SGD", "Adam"]:
        possible = ["StepLR", "MultiStepLR", "OneCycleLR", "CosineAnnealingLR"]
    else:
        possible = ["ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingWarmRestarts"]
    if epochs < 10 and "CosineAnnealingWarmRestarts" in possible:
        possible.remove("CosineAnnealingWarmRestarts")
    return random.choice(possible)

def choose_metrics(num_classes):
    all_metrics = ["Accuracy", "F1", "Jaccard", "Precision", "Recall"]
    if num_classes <= 2:
        # Restreint aux métriques adaptées au binaire
        metrics = ["F1", "Jaccard", "Precision", "Recall"]
    else:
        # Tous les choix possibles
        metrics = all_metrics

    k = random.randint(1, len(metrics))
    return random.sample(metrics, k)
    
def fetch_dynamic_available_opts():
        
        url_map = {
            "UnetSegmentor": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/UnetSegmentor.md",
            "UnetVanilla": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/UnetVanilla.md",
            "DINOv2": "https://raw.githubusercontent.com/FloFive/SCHISM/main/docs/DINOv2.md"
        }
        
        # === PARAMÈTRES PAR MODÈLE (docs séparées) ===
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
        return model_params

def build_model_config(model_type: str, num_classes: int, model_params: dict) -> dict:
    """
    Construit dynamiquement config["Model"] en fonction des paramètres
    disponibles pour le modèle choisi.
    """
    # base minimale
    config_model = {
        "model_type": model_type,
        "num_classes": str(num_classes),
    }

    # paramètres documentés
    for param in model_params.get(model_type, []):
        if param in config_model:  # déjà défini
            continue

        # valeurs par défaut simples
        defaults = {
            "n_block": "4",
            "k_size": "3",
            "activation": "leakyrelu",
            "channels": "8",
            "p": "0.5"
        }
        config_model[param] = defaults.get(param, None)  # None si inconnu

    return config_model
#==============================================================================

for i in range(1, nb_batch + 1):
    sub_f = final_configs_dir / f"config_{i:03d}"
    sub_f.mkdir(exist_ok=True)
    hyper_file = sub_f / "hyperparameters.ini"

    config = configparser.ConfigParser()

    model_params = fetch_dynamic_available_opts()
    model_type = random.choice(model_types)
    optimizer = random.choice(optimizers)
    epochs = 15
    num_classes = random.choice([2, 3])
    activation = random.choice(activations)

    loss = choose_loss(num_classes)
    scheduler = choose_scheduler(optimizer, epochs)
    metric = choose_metrics(num_classes)

    img_res = random.choice([256, 512])
    crop_size = random.choice([128, 192, 224]) if img_res == 256 else random.choice([224, 384, 448])

    config["Model"] = stringify_dict(build_model_config(model_type, num_classes, model_params))

    config["Optimizer"] = {"optimizer": optimizer}
    config["Optimizer"].update(stringify_dict(optimizer_extra_args.get(optimizer, {"lr": 0.001})))

    config["Scheduler"] = {"scheduler": scheduler}
    config["Scheduler"].update(stringify_dict(scheduler_extra_args.get(scheduler, {})))

    config["Loss"] = {"loss": loss, "ignore_background": "True", "weights": "True"}

    config["Training"] = {
        "batch_size": str(random.choice([4, 8, 16])),
        "val_split": str(random.choice([0.1, 0.2, 0.3])),
        "epochs": str(epochs),
        "metrics": ", ".join(metric),
        "early_stopping": "True"
    }

    config["Data"] = {
        "crop_size": str(crop_size),
        "img_res": str(img_res),
        "num_samples": "500"
    }

    with open(hyper_file, "w", encoding="utf-8") as f:
        config.write(f)

print("Configuration files created successfully.\n")

#==============================================================================

# select = answer_yes_or_no("Creating a batch file for training")

# if select:
#     data_source = input(f"[?] Enter the data directory: ").strip()
#     output_file = final_configs_dir / 'batch_multi.txt'

#     with open(output_file, 'w', encoding="utf-8") as f:
#         for i in range(1, nb_batch + 1):
#             config_number = f"{i:03}"
#             f.write("2\n")
#             f.write(f"{data_source}\n")
#             config_path = final_configs_dir / f"config_{config_number}"
#             f.write(f"{config_path}\n")
#         f.write("4\ny\n")

#     print(f"\nThe file '{output_file}' created successfully.")
