"""
Created by Analitika at 03/07/2024
contact@analitika.fr
"""

# External imports
import typer

# import wandb
from datasets import load_from_disk
from loguru import logger
from trl import SFTConfig, SFTTrainer

import wandb

# Internal imports
from local_llama3.config import MODEL_DATA_DIR, PROCESSED_DATA_MED_DIR, wandb_login
from models.load_model import load_model_tokenizer

app = typer.Typer()

MODEL_NAME = "fb_opt_350m"


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_MED_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_MED_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(f"Training {MODEL_NAME} model here...")
    wandb_login()
    wandb.init(project="Fine-tune Medical Dataset", job_type="training", anonymous="allow")
    # Load the tokenizer and then set up a model and tokenizer for conversational AI tasks.
    model, tokenizer = load_model_tokenizer(MODEL_NAME)

    # Adding the adapter to the layer
    # To improve the training time, we’ll attach the adapter layer with a few parameters,
    # making the entire process faster and more memory-efficient.
    # LoRA config:
    # • LoraConfig is a class that takes several parameters to configure the model.
    # • r=16 sets the reduction factor of the model.
    # • lora_alpha=32 sets the scale factor of the model.
    # • lora_dropout=0.05 sets the dropout rate, which is a regularization technique to prevent overfitting.
    # • bias="none" indicates that no bias is used in the model.
    # • task_type="CAUSAL_LM" sets the task type to causal language modeling.
    # • target_modules is a list of modules that the model will target.
    # • get_peft_model(model, peft_config) is a function that takes the model and the configuration as arguments.
    # • model = get_peft_model(model, peft_config) assigns the configured model to the variable model.
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    # )
    # model = get_peft_model(model, peft_config)

    # Loading the dataset
    storage_folder = str(PROCESSED_DATA_MED_DIR).replace("medical", "fb_medical")
    dataset = load_from_disk(storage_folder)
    dataset = dataset.train_test_split(test_size=0.1)
    # print(dataset["text"][3])
    output_dir = str(MODEL_DATA_DIR).replace("medical", "fb_medical")

    sft_config = SFTConfig(
        dataset_text_field="text",  # Name of the field in the dataset that contains the text to be processed.
        max_seq_length=512,  # Maximum sequence length for the model's input data.
        output_dir=output_dir,  # Directory where the model's output and checkpoints will be saved.
        per_device_train_batch_size=1,  # Batch size for training on each device (GPU/CPU).
        per_device_eval_batch_size=1,  # Batch size for evaluation on each device (GPU/CPU).
        gradient_accumulation_steps=2,  # Number of steps to accumulate gradients before updating the model's weights.
        optim="paged_adamw_32bit",  # Optimizer to use for training. Here, it's a 32-bit AdamW optimizer.
        num_train_epochs=1,  # Number of epochs to train the model.
        eval_strategy="steps",  # Evaluation strategy to use, e.g., evaluate after a certain number of steps.
        eval_steps=0.2,
        # Number of steps between evaluations during training. Here, 0.2 means evaluation occurs every 20% of the steps.
        logging_steps=1,  # Number of steps between logging updates.
        warmup_steps=10,  # Number of steps to perform learning rate warmup.
        logging_strategy="steps",  # Logging strategy, e.g., log after a certain number of steps.
        learning_rate=2e-4,  # Learning rate for training.
        fp16=False,  # Use 16-bit floating point precision (half precision) during training if True.
        bf16=False,  # Use bfloat16 precision during training if True.
        group_by_length=True,  # Group sequences of similar lengths together during training to improve efficiency.
        packing=False,  # Whether to use example packing to combine multiple sequences into a single input sequence.
        report_to="wandb",  # Tool to report training progress to (e.g., Weights and Biases).
    )

    # We set up a supervised fine-tuning (SFT) trainer and provide a train and evaluation dataset,

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=sft_config,
    )

    trainer.train()

    # Model evaluation
    # When you finish the Weights & Biases session, it’ll generate the run history and summary.

    wandb.finish()
    model.config.use_cache = True


if __name__ == "__main__":
    app()
