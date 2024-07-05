"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

# import os

# External imports
# from pathlib import Path

# import torch
import typer

# import wandb
# from datasets import load_dataset, load_from_disk
from loguru import logger

# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Internal imports
# from local_llama3.config import PROCESSED_DATA_MED_DIR

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     TrainingArguments,
#     logging,
#     pipeline,
# )
# from trl import SFTTrainer, setup_chat_format


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_MED_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_MED_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # dataset = load_from_disk(str(PROCESSED_DATA_MED_DIR))
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model here...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


def explore():
    pass
    # dataset = load_from_disk(str(PROCESSED_DATA_MED_DIR))


if __name__ == "__main__":
    app()
