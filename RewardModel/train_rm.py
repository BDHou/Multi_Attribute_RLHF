import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the HH-RLHF comparison dataset
dataset = load_dataset("Dahoas/full-hh-rlhf")

# Flatten each comparison into two examples (prompt+chosen and prompt+rejected)
def flatten_comparisons(batch):
    texts = []
    labels = []
    for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
        texts.append(prompt + chosen)
        labels.append(1)  # label 1 for chosen (preferred)
        texts.append(prompt + rejected)
        labels.append(0)  # label 0 for rejected
    return {"text": texts, "label": labels}

# Apply flattening to training and test splits
train_data = dataset["train"].map(flatten_comparisons, batched=True, remove_columns=dataset["train"].column_names)
test_data  = dataset["test"].map(flatten_comparisons,  batched=True, remove_columns=dataset["test"].column_names)

# Load the tokenizer and model (Open-LLaMA 3B with a classification head)
model_name = "weqweasdas/hh_rlhf_rm_open_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,                # Set up for binary classification (preferred vs rejected)
    ignore_mismatched_sizes=True # Ignore head shape mismatch and initialize new head
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Filter out examples that are too long (>512 tokens)
MAX_LENGTH = 512
def too_long(example):
    return len(tokenizer(example["text"], truncation=False)["input_ids"]) > MAX_LENGTH

train_data = train_data.filter(lambda ex: not too_long(ex))
test_data  = test_data.filter(lambda ex: not too_long(ex))

# Tokenize texts with padding/truncation
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_data = train_data.map(tokenize, batched=True, remove_columns=["text"])
test_data  = test_data.map(tokenize,  batched=True, remove_columns=["text"])

# Rename label column to "labels" as required by the Trainer
train_data = train_data.rename_column("label", "labels")
test_data  = test_data.rename_column("label", "labels")

# (Removed manual float casting of labels to keep them as integers for cross-entropy)

# Set the dataset format to PyTorch tensors
train_data.set_format("torch")
test_data.set_format("torch")

# Define evaluation metric (accuracy)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    output_dir="rm_model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_steps=100,
    fp16=True,  # use FP16 if available for speed
    # bf16=True can be used on supported GPUs (e.g., A100), if preferred
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate(eval_dataset=test_data)
print(f"Test accuracy: {eval_results['eval_accuracy']:.2%}")