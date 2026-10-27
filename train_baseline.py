
from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Column names used in your dataset.
# Change TEXT_COL if your CSV uses another text column name.
TEXT_COL = "text"
CATEGORY_COL = "category"
PRIORITY_COL = "priority"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = [
        TEXT_COL,
        CATEGORY_COL,
        PRIORITY_COL,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Update TEXT_COL, CATEGORY_COL, and PRIORITY_COL "
            "at the top of this script."
        )

    df = df.dropna(subset=required_columns).copy()

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[df[TEXT_COL] != ""].copy()

    return df


# --------------------------------------------------
# 3. WORD-LEVEL TF-IDF PIPELINE
# --------------------------------------------------

def create_word_pipeline():

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# 4. WORD + CHARACTER TF-IDF PIPELINE (BONUS)
# --------------------------------------------------

def create_combined_pipeline():

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                lowercase=True,
                ngram_range=(3, 5),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# 5. HYPERPARAMETER TUNING WITH CROSS-VALIDATION
# --------------------------------------------------

def tune_model(X_train, y_train, target, feature_type):

    # Use at most five folds, and ensure each class
    # has enough examples for stratified cross-validation.
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError(
            f"Not enough training examples per class for {target}."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if feature_type == "word":

        pipeline = create_word_pipeline()

        param_grid = {
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2),
                (1, 3),
            ],
            "classifier__C": [
                0.1,
                1,
                10,
            ],
        }

    else:

        pipeline = create_combined_pipeline()

        param_grid = {
            "features__word__ngram_range": [
                (1, 1),
                (1, 2),
            ],
            "features__char__ngram_range": [
                (3, 5),
                (3, 6),
            ],
            "classifier__C": [
                0.1,
                1,
                10,
            ],
        }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("\n" + "=" * 60)
    print(f"TUNING: {target.upper()} | {feature_type}")
    print("=" * 60)

    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best CV macro F1:", round(search.best_score_, 4))

    return search.best_estimator_


# --------------------------------------------------
# 6. EVALUATION AND CONFUSION MATRIX
# --------------------------------------------------

def evaluate_model(model, X_val, y_val, target, feature_type):

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS: {target.upper()} | {feature_type}")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    labels = model.classes_

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=labels,
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(
        f"{target.title()} - {feature_type} Confusion Matrix"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    matrix_path = (
        RESULT_DIR /
        f"{target}_{feature_type}_confusion_matrix.png"
    )

    plt.savefig(matrix_path, dpi=150)
    plt.close()

    print("Confusion matrix saved:", matrix_path)

    # --------------------------------------------------
    # 7. FIND THE 10 MOST CONFUSING VALIDATION EXAMPLES
    # --------------------------------------------------

    probabilities = model.predict_proba(X_val)

    errors = []

    for index, (text, actual, predicted) in enumerate(
        zip(X_val.tolist(), y_val.tolist(), predictions)
    ):

        if actual == predicted:
            continue

        probs = probabilities[index]
        ranked_indices = probs.argsort()[::-1]

        confidence = float(probs[ranked_indices[0]])

        if len(ranked_indices) > 1:
            margin = float(
                probs[ranked_indices[0]]
                - probs[ranked_indices[1]]
            )
        else:
            margin = confidence

        errors.append({
            "ticket_text": text,
            "actual_label": actual,
            "predicted_label": predicted,
            "confidence": round(confidence, 4),
            "confidence_margin": round(margin, 4),
            "possible_reason": (
                "Review for ambiguous wording, overlapping labels, "
                "missing context, or class imbalance."
            ),
        })

    error_columns = [
        "ticket_text",
        "actual_label",
        "predicted_label",
        "confidence",
        "confidence_margin",
        "possible_reason",
    ]

    error_df = pd.DataFrame(errors, columns=error_columns)

    if not error_df.empty:
        error_df = error_df.sort_values(
            "confidence_margin",
            ascending=True,
        ).head(10)

    error_path = (
        RESULT_DIR /
        f"{target}_{feature_type}_errors.csv"
    )

    error_df.to_csv(error_path, index=False)

    print("Error analysis saved:", error_path)
    print(f"Misclassified examples exported: {len(error_df)}")

    return {
        "task": target,
        "features": feature_type,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix_path.name,
        "error_file": error_path.name,
    }


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train_df = load_data("train.csv")
    val_df = load_data("validation.csv")

    print("\nTraining rows:", len(train_df))
    print("Validation rows:", len(val_df))

    print("\nTraining category distribution:")
    print(train_df[CATEGORY_COL].value_counts().to_string())

    print("\nTraining priority distribution:")
    print(train_df[PRIORITY_COL].value_counts().to_string())

    X_train = train_df[TEXT_COL]
    X_val = val_df[TEXT_COL]

    tasks = {
        "category": CATEGORY_COL,
        "priority": PRIORITY_COL,
    }

    all_results = []

    for target, label_column in tasks.items():

        y_train = train_df[label_column]
        y_val = val_df[label_column]

        unseen_labels = set(y_val.unique()) - set(y_train.unique())

        if unseen_labels:
            raise ValueError(
                f"Validation contains labels not seen in training "
                f"for {target}: {unseen_labels}"
            )

        for feature_type in ["word", "word_char"]:

            model = tune_model(
                X_train,
                y_train,
                target,
                feature_type,
            )

            metrics = evaluate_model(
                model,
                X_val,
                y_val,
                target,
                feature_type,
            )

            all_results.append(metrics)

            model_path = (
                MODEL_DIR /
                f"{target}_{feature_type}.joblib"
            )

            joblib.dump(model, model_path)

            print("Model saved:", model_path)

    # --------------------------------------------------
    # 9. SAVE METRICS AND COMPARISON TABLE
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    metrics_csv = RESULT_DIR / "metrics.csv"
    results_df.to_csv(metrics_csv, index=False)

    summary = {}

    for result in all_results:
        key = f"{result['task']}_{result['features']}"

        summary[key] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
        }

    metrics_json = RESULT_DIR / "metrics.json"

    with open(metrics_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 60)

    print(
        results_df[
            ["task", "features", "accuracy", "macro_f1"]
        ].to_string(index=False)
    )

    print("\nMetrics saved:", metrics_csv)
    print("JSON summary saved:", metrics_json)
    print("Models directory:", MODEL_DIR)
    print("Results directory:", RESULT_DIR)

    print("\nTASK 2 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
