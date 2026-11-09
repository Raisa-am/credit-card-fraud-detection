"""
Credit Card Fraud Detection
---------------------------
Run:
    pip install -r requirements.txt
    python fraud_detection.py

If a Kaggle creditcard.csv file is present in the project folder, the script uses it.
Otherwise, it creates clearly-labelled synthetic demo data so the project can run
without downloading a dataset. Synthetic results are for demonstration only.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)
DATA_FILE = ROOT / "creditcard.csv"
RANDOM_STATE = 42


def load_data():
    """Load Kaggle data when present; otherwise create synthetic demo data."""
    if DATA_FILE.exists():
        print(f"Loading real dataset: {DATA_FILE.name}")
        df = pd.read_csv(DATA_FILE)
        if "Class" not in df.columns:
            raise ValueError("Dataset must contain a 'Class' target column (0=legitimate, 1=fraud).")
        source = "Kaggle creditcard.csv"
    else:
        print("creditcard.csv not found. Creating synthetic demonstration data.")
        print("IMPORTANT: These are simulated records, not real credit-card transactions.")
        X, y = make_classification(
            n_samples=12000,
            n_features=30,
            n_informative=12,
            n_redundant=8,
            n_classes=2,
            weights=[0.985, 0.015],
            class_sep=1.2,
            random_state=RANDOM_STATE,
        )
        columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        df = pd.DataFrame(X, columns=columns)
        df["Amount"] = (df["Amount"].abs() * 100).round(2)
        df["Time"] = (df["Time"].abs() * 1000).round(0)
        df["Class"] = y
        source = "Synthetic demonstration data"
    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)
    return df, source


def main():
    df, source = load_data()
    if df["Class"].nunique() != 2:
        raise ValueError("The target column must contain both classes: 0 and 1.")

    print("\nDataset source:", source)
    print("Rows:", len(df))
    print("Fraud cases:", int(df["Class"].sum()))
    print("Fraud percentage:", round(df["Class"].mean() * 100, 4), "%")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    X = X.select_dtypes(include=["number"]).replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling supports Logistic Regression. Random Forest is trained on the same
    # scaled inputs to keep the workflow easy to compare.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Precision (fraud)": report["1"]["precision"],
            "Recall (fraud)": report["1"]["recall"],
            "F1-score (fraud)": report["1"]["f1-score"],
            "ROC-AUC": roc_auc,
            "Average precision": avg_precision,
        })

        print(classification_report(y_test, predictions, zero_division=0))
        print(f"ROC-AUC: {roc_auc:.4f} | Average precision: {avg_precision:.4f}")

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["Legitimate", "Fraud"],
            cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(f"{name} — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    results_df = pd.DataFrame(results).sort_values("F1-score (fraud)", ascending=False)
    results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # Save a simple class distribution chart.
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values)
    ax.set_title("Transaction Class Distribution")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(OUTPUT / "class_distribution.png", dpi=160)
    plt.close(fig)

    print(f"\nFinished. Results and charts are saved in: {OUTPUT}")
    if source.startswith("Synthetic"):
        print("Reminder: Replace the synthetic demo with the Kaggle dataset before reporting real project results.")


if __name__ == "__main__":
    main()
