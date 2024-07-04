"""
Created by Analitika at 04/07/2024
contact@analitika.fr
"""

# External imports
import os

# import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format

# Internal imports
from local_llama3.config import (
    ATTN_IMPLEMENTATION,
    BASE_MODEL,
    MODEL_DATA_DIR,
    TOKENIZER_DATA_DIR,
    TORCH_DTYPE,
)


# Function to check if the model exists locally
def model_exists_locally(model_dir):
    required_files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]
    return all((model_dir / file).exists() for file in required_files)


def tokenizer_exists_locally(tokenizer_dir):
    required_files = [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    return all((tokenizer_dir / file).exists() for file in required_files)


# Function to load the model
def load_model(model_dir):
    if model_exists_locally(model_dir):
        logger.info(f"Loading model from local directory: {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
        )
    else:
        logger.info(f"Model not found locally. Downloading from Hugging Face: {BASE_MODEL}")
        # Define the quantization configuration
        # In this project we reduce memory usage and speed up the fine-tuning process.
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
        )
        """
        Downloading shards: 100%|██████████| 4/4 [1:46:51<00:00, 1602.78s/it]
        Loading checkpoint shards: 100%|██████████| 4/4 [00:46<00:00, 11.63s/it]
        """
        logger.success("Model successfully downloaded from Hugging Face")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(str(model_dir))
        logger.info(f"Model saved to {model_dir}")
    return model


# Function to load the tokenizer
def load_model_tokenizer(subfolder="llama3"):
    # Load the model
    model_ = load_model(MODEL_DATA_DIR / subfolder)
    tokenizer_dir = TOKENIZER_DATA_DIR / subfolder
    if tokenizer_exists_locally(tokenizer_dir):
        logger.info(f"Loading tokenizer from local directory: {tokenizer_dir}")
        tokenizer_ = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        logger.info(f"Tokenizer not found locally. Downloading from Hugging Face: {BASE_MODEL}")
        tokenizer_ = AutoTokenizer.from_pretrained(BASE_MODEL)
        logger.success("Tokenizer successfully downloaded from Hugging Face")
        model_, tokenizer_ = setup_chat_format(model_, tokenizer_)
        tokenizer_.save_pretrained(str(tokenizer_dir))
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_.save_pretrained(str(tokenizer_dir))
        logger.info(f"Tokenizer saved to {tokenizer_dir}")
    return model_, tokenizer_


if __name__ == "__main__":
    # Load the tokenizer
    model, tokenizer = load_model_tokenizer()

    # Determine the device and move the inputs to it
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.info(f"Detected device: {device}.")

    # Test the model and tokenizer
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    # inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # Move inputs to the same device as the model
    # inputs = {key: value.to(device) for key, value in inputs.items()}
    logger.info("Calling to the model, check results")
    outputs = model.generate(**inputs)
    logger.info("Result from model obtained")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.success(response)
