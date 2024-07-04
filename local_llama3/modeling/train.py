"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

# External imports
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os
import torch
import wandb
from trl import SFTTrainer, setup_chat_format
import typer
from loguru import logger
from tqdm import tqdm

# Internal imports
from local_llama3.config import MODELS_DIR, PROCESSED_DATA_MED_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_MED_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_MED_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    dataset = load_from_disk(str(PROCESSED_DATA_MED_DIR))
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


def explore():
    dataset = load_from_disk(str(PROCESSED_DATA_MED_DIR))


if __name__ == "__main__":
    app()
