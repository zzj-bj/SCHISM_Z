import logging
import nest_asyncio
import openai
import os

# ================================
# Configuration
# ================================

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
# project_dir = 'C:/Users/Mehdi/Documents/Code/Akkodis/SCHISM/'
# root_folder = os.path.join(project_dir, "runs", "knowledge_base", "augmented_configs")

nest_asyncio.apply()

# ⚠️ Configure ton client OpenAI (adapte selon ton serveur/API)
client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

# External URLs for fetching detailed documentation
EXTERNAL_URLS = {
    'Optimizer': {
        'Adagrad': "https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html",
        'Adam': "https://pytorch.org/docs/stable/generated/torch.optim.Adam.html",
        'AdamW': "https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
        'NAdam': "https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html",
        'RMSprop': "https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html",
        'RAdam': "https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html",
        'SGD': "https://pytorch.org/docs/stable/generated/torch.optim.SGD.html"
    },
    'Scheduler': {
        "LRScheduler": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html",
        "LambdaLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html",
        "MultiplicativeLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html",
        "StepLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html",
        "MultiStepLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html",
        "ConstantLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html",
        "LinearLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html",
        "ExponentialLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html",
        "PolynomialLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html",
        "CosineAnnealingLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html",
        "SequentialLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html",
        "ReduceLROnPlateau": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html",
        "CyclicLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html",
        "OneCycleLR": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html",
        "CosineAnnealingWarmRestarts": "https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html"
    },
    'Loss': {
        'CrossEntropyLoss': "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
        'BCEWithLogitsLoss': "https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html",
        'NLLLoss': "https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html"
    }
}