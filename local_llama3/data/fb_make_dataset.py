"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

import shutil

# External imports
from pathlib import Path

import typer

# from transformers import AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from loguru import logger

# Internal imports
from local_llama3.config import DATASET_NAME, PROCESSED_DATA_MED_DIR, RAW_DATA_MED_DIR
from models.load_model import load_model_tokenizer

# from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


app = typer.Typer()


def format_chat_template(row, tokenizer):
    row_json = [
        {"role": "user", "content": row["Patient"]},
        {"role": "assistant", "content": row["Doctor"]},
    ]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


def dataset_exists_locally(dataset_dir):
    required_files = [
        "dataset_info.json",
        "state.json",
    ]
    return all((dataset_dir / file).exists() for file in required_files)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_MED_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_MED_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # Define storage directory
    storage_folder = str(PROCESSED_DATA_MED_DIR).replace("medical", "fb_medical")
    # Define a custom cache directory
    cache_dir = Path(RAW_DATA_MED_DIR) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Load the dataset
    if dataset_exists_locally(RAW_DATA_MED_DIR):
        logger.info(f"Loading dataset from local directory: {RAW_DATA_MED_DIR}")
        dataset = load_from_disk(str(RAW_DATA_MED_DIR))
    else:
        logger.info("Downloading dataset...")
        dataset = load_dataset(DATASET_NAME, split="all", cache_dir=str(cache_dir))
        logger.info("Dataset downloaded.")
        # Ensure the RAW_DATA_MED_DIR exists
        Path(RAW_DATA_MED_DIR).mkdir(parents=True, exist_ok=True)
        # Save the dataset to the data/raw directory
        dataset.save_to_disk(str(RAW_DATA_MED_DIR))
        logger.success(f"Raw dataset saved to {RAW_DATA_MED_DIR}")

    logger.success("Raw dataset loaded")

    if not dataset_exists_locally(Path(storage_folder)):
        # Preprocess the dataset
        # preproc_dataset = dataset.shuffle(seed=65).select(range(1000))
        _, tokenizer = load_model_tokenizer("fb_opt_350m")
        # Preprocess the dataset
        preproc_dataset = dataset.map(
            lambda row: format_chat_template(row, tokenizer),
            num_proc=4,
            cache_file_name=str(cache_dir / "processed_dataset.arrow"),
        )
        Path(storage_folder).mkdir(parents=True, exist_ok=True)
        preproc_dataset.save_to_disk(storage_folder)
        logger.success(f"Preprocessed dataset saved to {storage_folder}")
        # Delete the cache directory
        shutil.rmtree(cache_dir)
        logger.info(f"Cache directory {cache_dir} deleted.")

    logger.success("Datasets generated")

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Processing dataset...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Processing dataset complete.")
    # -----------------------------------------


def explore():
    storage_folder = str(PROCESSED_DATA_MED_DIR).replace("medical", "fb_medical")
    dataset = load_from_disk(storage_folder)
    print(dataset["text"][3])


if __name__ == "__main__":
    app()

    """
    dataset = load_dataset("imdb", split="train")
    dataset["text"][0]
    'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was
    first released in 1967. ...
    """

    """
    dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
    dataset["instruction"][0]
    'Create a function that takes a specific input and produces a specific output using any mathematical operators.
    Write corresponding code in Python.'
    dataset["output"][0]: code python
    """
