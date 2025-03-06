from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import (
    Dataset,
    DatasetDict,
    Sequence,
    ClassLabel,
    load_dataset
)
from sklearn.metrics import (
    accuracy_score
)
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="The local path of the model.")
parser.add_argument("--batch_size", type=int, help="The batch size.", default=16)

args = parser.parse_args()


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

datasets_path = {
    "train": os.path.join(BASE_PATH, "data/train-00000-of-00001.parquet"),
    "validation": os.path.join(BASE_PATH, "data/validation-00000-of-00001.parquet"),
    "test": os.path.join(BASE_PATH, "data/test-00000-of-00001.parquet")
}

ds = load_dataset("parquet", data_files=datasets_path)

ds = DatasetDict({
    split: ds[split].filter(lambda example: example["label"] != -1)
    for split in ds.keys()
})

tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(BASE_PATH, "model/roberta-base"))

def tokenize_function(examples):
    # Tokenize premise and hypothesis with padding and truncation
    tokenized_inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=True
    )
    return tokenized_inputs

# Apply tokenization to all splits
tokenizer_ds = DatasetDict({
    split: ds[split].map(
        tokenize_function,
        batched=True,
        remove_columns=["premise", "hypothesis"]
    ).rename_column("label", "labels")
    for split in ds.keys()
})

def compute_metrics(p):

    predictions, labels = p

    # We run an argmax to get the label
    predictions = np.argmax(predictions, axis=-1)


    mec = {
        "accuracy": accuracy_score(labels, predictions)
    }

    return mec




batch_size = args.batch_size
RUN_ID = "RoBERTa-ner"
SEED = 0
LR = 1e-6

train_args = TrainingArguments(
    f"{RUN_ID}_{SEED}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=batch_size,
    num_train_epochs=100,
    warmup_ratio=0,
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    metric_for_best_model="accuracy",
    logging_strategy="epoch",
    seed=SEED,
)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=3, torch_dtype="auto")

for param in model.roberta.parameters():
    param.requires_grad = False

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenizer_ds["train"],
    eval_dataset=tokenizer_ds["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# torch.autograd.set_detect_anomaly(True)

trainer.train()