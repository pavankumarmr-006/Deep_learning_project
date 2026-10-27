
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "ticket_text"
TARGETS = ["category", "priority"]

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = json.loads(
    (REPORT_DIR / "distilbert_results.json").read_text()
)

rows = []

for target in TARGETS:
    baseline_path = MODEL_DIR / f"{target}_baseline.joblib"
    if not baseline_path.exists():
        print(f"Skipping missing baseline: {baseline_path}")
        continue

    baseline = joblib.load(baseline_path)
    bert_dir = MODEL_DIR / f"distilbert_{target}"

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_dir
    ).to(device)
    model.eval()

    # Both models are evaluated on identical rows.
    texts = test_df[TEXT_COL].fillna("").astype(str).tolist()
    y_true = test_df[target].astype(str).to_numpy()

    # Baseline inference latency
    for _ in range(10):
        baseline.predict(texts[:1])

    n_runs = 5
    start = time.perf_counter()
    for _ in range(n_runs):
        baseline.predict(texts)
    baseline_total_seconds = (
        time.perf_counter() - start
    ) / n_runs

    baseline_pred = baseline.predict(texts)
    baseline_ms = baseline_total_seconds * 1000 / len(texts)

    # DistilBERT predictions in batches
    bert_predictions = []
    start = time.perf_counter()

    batch_size = 32
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[i:i + batch_size],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(device)

            logits = model(**batch).logits
            ids = torch.argmax(logits, dim=-1).cpu().numpy()

            bert_predictions.extend(
                model.config.id2label[int(idx)] for idx in ids
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    bert_ms = (
        time.perf_counter() - start
    ) * 1000 / len(texts)

    # Evaluate baseline using the test labels.
    from sklearn.metrics import accuracy_score, f1_score

    rows.extend([
        {
            "Task": target,
            "Model": "TF-IDF + Logistic Regression",
            "Accuracy": accuracy_score(y_true, baseline_pred),
            "Macro F1": f1_score(
                y_true, baseline_pred,
                average="macro", zero_division=0
            ),
            "Training time (s)": np.nan,
            "Inference time/ticket (ms)": baseline_ms
        },
        {
            "Task": target,
            "Model": "Fine-tuned DistilBERT",
            "Accuracy": accuracy_score(y_true, bert_predictions),
            "Macro F1": f1_score(
                y_true, bert_predictions,
                average="macro", zero_division=0
            ),
            "Training time (s)": results[target]["training_seconds"],
            "Inference time/ticket (ms)": bert_ms
        }
    ])

comparison = pd.DataFrame(rows)
comparison.to_csv(
    REPORT_DIR / "model_comparison.csv", index=False
)

print(comparison.round(4).to_string(index=False))
print("\nSaved reports/model_comparison.csv")
