"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Authenticate using your Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Set the HUGGINGFACE_API_TOKEN environment variable.")
login(HUGGINGFACE_TOKEN)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
MODEL_DATA_DIR = DATA_DIR / "model"
TOKENIZER_DATA_DIR = DATA_DIR / "tokenizer"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_MED_DIR = RAW_DATA_DIR / "medical"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_MED_DIR = PROCESSED_DATA_DIR / "medical"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Set the base model, dataset, and new model variable.
# We’ll load the base model from Kaggle and the dataset
# from the HugginFace Hub and then save the new model.
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
# BASE_MODEL = "xlm-roberta-base"
DATASET_NAME = "ruslanmv/ai-medical-chatbot"
NEW_MODEL = "llama-3-8b-chat-doctor"

# Set the data type and attention implementation
TORCH_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "eager"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

if __name__ == "__main__":
    logger.info("Done")
