from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "CEAS_08.csv"
MODEL_PATH = PROJECT_DIR / "models" / "emailguard_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "reports" / "metrics.txt"
CONFUSION_PATH = PROJECT_DIR / "reports" / "confusion_matrix.png"

RANDOM_SEED = 42 
TEST_SIZE = 0.20 

POSITIVE_LABEL = 1
LABEL_NAMES = {0: "legitimate", 1: "phishing"}

UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65


def load_data(path):
    """Read the CSV and keep only the columns the model uses."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    df["text"] = df["subject"] + "\n" + df["body"]
    return df[["text", "label"]]


def clean_data(df):
    """
    Duplicates must be removed BEFORE the train/test split. Otherwise the same
    email could appear in both sets, and the test score would be too optimistic
    (a form of data leakage).
    """

    df = df.copy()
    df["key"] = df["text"].str.lower().str.split().str.join(" ")

    labels_per_key = df.groupby("key")["label"].nunique()
    conflicting_keys = labels_per_key[labels_per_key > 1].index
    n_conflicting_rows = df["key"].isin(conflicting_keys).sum()
    df = df[~df["key"].isin(conflicting_keys)]

    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first")
    n_duplicates = before - len(df)

    before = len(df)
    df = df[df["key"].str.len() > 0]
    n_empty = before - len(df)

    stats = {
        "conflicting_label_rows_removed": int(n_conflicting_rows),
        "conflicting_emails": int(len(conflicting_keys)),
        "duplicate_rows_removed": int(n_duplicates),
        "empty_rows_removed": int(n_empty),
        "rows_after_cleaning": int(len(df)),
    }
    return df[["text", "label"]].reset_index(drop=True), stats


def build_pipeline():
    """
    A Pipeline joins the two steps so that, when we call fit() on the training
    data, the TF-IDF vocabulary is learned from training emails only. The test
    emails are never seen during fitting.
    """
    vectoriser = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),   # single words and two-word phrases ("click here")
        min_df=2,             # ignore terms that appear in only one email
        max_df=0.95,          # ignore terms that appear in almost every email
        max_features=50_000,  # cap the vocabulary size to keep things fast
        sublinear_tf=True,    # stop very repeated words dominating
    )
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("tfidf", vectoriser), ("logreg", classifier)])


def top_features(pipeline, n=20):
    """
    Logistic regression gives every term a weight (coefficient).
    Positive weight pushes towards phishing; negative pushes towards legitimate.
    Looking at the strongest terms shows whether the model learned phishing
    language or just quirks of the dataset (names, dates, mailing lists).
    """
    terms = pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = pipeline.named_steps["logreg"].coef_[0]
    table = pd.DataFrame({"term": terms, "weight": weights})
    towards_phishing = table.sort_values("weight", ascending=False).head(n)
    towards_legit = table.sort_values("weight", ascending=True).head(n)
    return towards_phishing, towards_legit


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Download CEAS_08.csv and put it in "
            "the data folder (see data/README.md)."
        )

    raw = load_data(DATA_PATH)
    df, cleaning_stats = clean_data(raw)
    print("Cleaning:", cleaning_stats)
    print("Class counts after cleaning:", df["label"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    print(f"Training emails: {len(X_train):,}   Test emails: {len(X_test):,}")

    pipeline = build_pipeline()
    print("Training... (this can take a minute)")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    scores = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(POSITIVE_LABEL)]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    recall = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL)
    labels_order = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    report = classification_report(
        y_test, y_pred, labels=labels_order,
        target_names=[LABEL_NAMES[i] for i in labels_order], digits=4,
    )

    in_band = ((scores >= UNCERTAIN_LOW) & (scores <= UNCERTAIN_HIGH)).sum()
    towards_phishing, towards_legit = top_features(pipeline)

    METRICS_PATH.parent.mkdir(exist_ok=True)
    lines = [
        "EmailGuard-XAI - model evaluation (generated by train.py)",
        f"scikit-learn version: {sklearn.__version__}",
        f"Random seed: {RANDOM_SEED}   Test size: {TEST_SIZE}",
        f"Assumed labels: {LABEL_NAMES} (verify against dataset docs)",
        "",
        f"Rows loaded: {len(raw):,}",
        *(f"{k}: {v:,}" for k, v in cleaning_stats.items()),
        f"Training emails: {len(X_train):,}",
        f"Test emails: {len(X_test):,}",
        "",
        "Test-set results (positive class = phishing):",
        f"  Accuracy:  {accuracy:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall:    {recall:.4f}",
        f"  F1-score:  {f1:.4f}",
        "",
        "Confusion matrix (rows = true label, columns = predicted label):",
        "                 pred legit   pred phishing",
        f"  true legit     {cm[0, 0]:>10}   {cm[0, 1]:>13}",
        f"  true phishing  {cm[1, 0]:>10}   {cm[1, 1]:>13}",
        "",
        "Classification report:",
        report,
        f"Test emails with model score between {UNCERTAIN_LOW} and {UNCERTAIN_HIGH} "
        f"(provisional uncertain band): {in_band:,} of {len(X_test):,}",
        "",
        "Top terms pushing towards PHISHING:",
        *(f"  {r.term!r}: {r.weight:.3f}" for r in towards_phishing.itertuples()),
        "",
        "Top terms pushing towards LEGITIMATE:",
        *(f"  {r.term!r}: {r.weight:.3f}" for r in towards_legit.itertuples()),
    ]
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")

    display = ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("EmailGuard-XAI test-set confusion matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150)
    plt.close()

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("\n".join(lines))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_PATH}")


if __name__ == "__main__":
    main()
