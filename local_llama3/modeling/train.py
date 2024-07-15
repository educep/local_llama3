# """
# Created by Analitika at 03/07/2024
# contact@analitika.fr
# """
#
# # import os
#
# # External imports
# # from pathlib import Path
#
# import os
#
# import typer
#
# # import wandb
# from datasets import load_from_disk
# from loguru import logger
# from peft import (  # PeftModel, prepare_model_for_kbit_training
#     LoraConfig,
#     get_peft_model,
# )
#
# # from tqdm import tqdm
# from transformers import (  # AutoModelForCausalLM,; AutoTokenizer,; HfArgumentParser,; logging,; pipeline,
#     TrainingArguments,
# )
# from trl import SFTConfig, SFTTrainer
#
# import wandb
#
# # Internal imports
# from local_llama3.config import MODEL_DATA_DIR, NEW_MODEL, PROCESSED_DATA_MED_DIR
# from models.load_model import load_model_tokenizer
#
# app = typer.Typer()
#
# MODEL_NAME = "fb_opt_350m"
#
#
# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     # features_path: Path = PROCESSED_DATA_MED_DIR / "features.csv",
#     # labels_path: Path = PROCESSED_DATA_MED_DIR / "labels.csv",
#     # model_path: Path = MODELS_DIR / "model.pkl",
#     # -----------------------------------------
# ):
#
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info(f"Training {MODEL_NAME} model here...")
#     wb_token = os.getenv("WANDB_TOKEN")
#     run = wandb.init(project="Fine-tune Medical Dataset", job_type="training", anonymous="allow")
#     # Load the tokenizer and then set up a model and tokenizer for conversational AI tasks.
#     model, tokenizer = load_model_tokenizer("fb_opt_350m")
#
#     # Adding the adapter to the layer
#     # To improve the training time, we’ll attach the adapter layer with a few parameters,
#     # making the entire process faster and more memory-efficient.
#     # LoRA config:
#     # • LoraConfig is a class that takes several parameters to configure the model.
#     # • r=16 sets the reduction factor of the model.
#     # • lora_alpha=32 sets the scale factor of the model.
#     # • lora_dropout=0.05 sets the dropout rate, which is a regularization technique to prevent overfitting.
#     # • bias="none" indicates that no bias is used in the model.
#     # • task_type="CAUSAL_LM" sets the task type to causal language modeling.
#     # • target_modules is a list of modules that the model will target.
#     # • get_peft_model(model, peft_config) is a function that takes the model and the configuration as arguments.
#     # • model = get_peft_model(model, peft_config) assigns the configured model to the variable model.
#     peft_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=[
#             "up_proj",
#             "down_proj",
#             "gate_proj",
#             "k_proj",
#             "q_proj",
#             "v_proj",
#             "o_proj",
#         ],
#     )
#     model = get_peft_model(model, peft_config)
#
#     # Loading the dataset
#     storage_folder = str(PROCESSED_DATA_MED_DIR).replace("medical", "fb_medical")
#     dataset = load_from_disk(storage_folder)
#     # print(dataset["text"][3])
#     dataset = dataset.train_test_split(test_size=0.1)
#
#     sft_config = SFTConfig(
#         dataset_text_field="text",
#         max_seq_length=512,
#         output_dir="/tmp",
#     )
#
#     training_arguments = TrainingArguments(
#         output_dir=NEW_MODEL,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=2,
#         optim="paged_adamw_32bit",
#         num_train_epochs=1,
#         evaluation_strategy="steps",
#         eval_steps=0.2,
#         logging_steps=1,
#         warmup_steps=10,
#         logging_strategy="steps",
#         learning_rate=2e-4,
#         fp16=False,
#         bf16=False,
#         group_by_length=True,
#         report_to="wandb",
#     )
#
#     # We set up a supervised fine-tuning (SFT) trainer and provide a train and evaluation dataset,
#     # LoRA configuration, training argument, tokenizer, and model.
#     # We keep the max_seq_length to 512 to avoid exceeding GPU memory during training.
#
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["test"],
#         peft_config=peft_config,
#         max_seq_length=512,
#         dataset_text_field="text",
#         tokenizer=tokenizer,
#         args=training_arguments,
#         packing=False,
#     )
#
#     # for i in tqdm(range(10), total=10):
#     #     if i == 5:
#     #         logger.info("Something happened for iteration 5.")
#     # logger.success("Modeling training complete.")
#     # -----------------------------------------
#
#
# def explore():
#     pass
#     # dataset = load_from_disk(str(PROCESSED_DATA_MED_DIR))
#
#
# if __name__ == "__main__":
#     app()
