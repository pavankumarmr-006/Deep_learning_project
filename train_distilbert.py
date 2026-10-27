
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "text"
TARGETS = ["category", "priority"]

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)

# --------------------------------------------------
# Load the SAME splits used in Part 1.
# Change filenames here to match your project.
# --------------------------------------------------
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

for df in [train_df, val_df, test_df]:
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()

if any(df[TEXT_COL].eq("").any() for df in
       [train_df, val_df, test_df]):
    raise ValueError("Empty ticket text found. Clean the data first.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

def tokenize_batch(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=256
    )

def prepare_dataset(df, label2id, target):
    temp = df[[TEXT_COL, target]].copy()
    temp = temp.rename(columns={target: "label"})
    temp["label"] = temp["label"].map(label2id)

    if temp["label"].isna().any():
        raise ValueError(
            f"Unknown or missing labels found for {target}"
        )

    ds = Dataset.from_pandas(
        temp.reset_index(drop=True),
        preserve_index=False
    )

    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.remove_columns([TEXT_COL])
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions,
            average="macro", zero_division=0
        )
    }

all_results = {}

# --------------------------------------------------
# Train one classifier per target
# --------------------------------------------------
for target in TARGETS:
    print(f"\n{'=' * 55}")
    print(f"Training DistilBERT for: {target}")
    print(f"{'=' * 55}")

    # Derive label mappings from training data only.
    classes = sorted(train_df[target].dropna().unique())
    label2id = {str(label): i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}

    if train_df[target].isna().any():
        raise ValueError(f"Missing training labels for {target}")

    for split_name, df in [
        ("validation", val_df),
        ("test", test_df)
    ]:
        unknown = set(df[target].dropna().astype(str)) - set(label2id)
        if unknown:
            raise ValueError(
                f"Unseen {target} labels in {split_name}: {unknown}"
            )
        if df[target].isna().any():
            raise ValueError(
                f"Missing {target} labels in {split_name}"
            )

    train_ds = prepare_dataset(train_df, label2id, target)
    val_ds = prepare_dataset(val_df, label2id, target)
    test_ds = prepare_dataset(test_df, label2id, target)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        label2id=label2id,
        id2label=id2label
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / f"{target}_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        report_to="none",
        seed=42,
        data_seed=42,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    start = time.perf_counter()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start

    # Validation evaluation
    val_metrics = trainer.evaluate(
        eval_dataset=val_ds,
        metric_key_prefix="val"
    )

    # Final test evaluation, after model selection
    test_output = trainer.predict(
        test_ds,
        metric_key_prefix="test"
    )

    y_true = test_output.label_ids
    y_pred = np.argmax(test_output.predictions, axis=-1)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(
        y_true, y_pred,
        average="macro", zero_division=0
    )

    # Classification report and confusion matrix
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=[str(c) for c in classes],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(classes)))
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    ).to_csv(REPORT_DIR / f"{target}_distilbert_confusion.csv")

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / f"{target}_distilbert_report.csv"
    )

    # Measure single-ticket latency on the same device.
    trainer.model.eval()
    sample = test_df[TEXT_COL].iloc[0]
    inputs = tokenizer(
        sample,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    trained_model = trainer.model.to(device)

    with torch.inference_mode():
        for _ in range(10):
            trained_model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    runs = 100
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(runs):
            trained_model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_ms = (
        (time.perf_counter() - start) / runs
    ) * 1000

    # Save model, tokenizer, and label mappings
    save_dir = MODEL_DIR / f"distilbert_{target}"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    with open(save_dir / "labels.json", "w") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {
                    str(k): v for k, v in id2label.items()
                }
            },
            f,
            indent=2
        )

    all_results[target] = {
        "validation_accuracy": val_metrics["val_accuracy"],
        "validation_macro_f1": val_metrics["val_macro_f1"],
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "training_seconds": training_seconds,
        "inference_ms_per_ticket": inference_ms
    }

    print(json.dumps(all_results[target], indent=2))

with open(REPORT_DIR / "distilbert_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nTraining complete. Models saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)
