"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

# import os

# External imports
from pathlib import Path

import typer
from datasets import load_dataset
from loguru import logger

# Internal imports
from local_llama3.config import DATASET_NAME, PROCESSED_DATA_MED_DIR, RAW_DATA_MED_DIR
from models.load_model import load_model_tokenizer

app = typer.Typer()


def format_chat_template(row):
    _, tokenizer = load_model_tokenizer()
    row_json = [
        {"role": "user", "content": row["Patient"]},
        {"role": "assistant", "content": row["Doctor"]},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_MED_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_MED_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # Load the dataset
    dataset = load_dataset(DATASET_NAME, split="all")

    # Ensure the RAW_DATA_MED_DIR exists
    Path(RAW_DATA_MED_DIR).mkdir(parents=True, exist_ok=True)
    # Save the dataset to the data/raw directory
    dataset.save_to_disk(str(RAW_DATA_MED_DIR))
    logger.success(f"Raw dataset saved to {RAW_DATA_MED_DIR}")

    preproc_dataset = dataset.shuffle(seed=65).select(
        range(1000)
    )  # Only use 1000 samples for quick demo
    preproc_dataset.save_to_disk(str(PROCESSED_DATA_MED_DIR))
    logger.success(f"Preprocessed dataset saved to {PROCESSED_DATA_MED_DIR}")

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
