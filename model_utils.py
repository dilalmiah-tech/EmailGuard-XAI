from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "emailguard_pipeline.joblib"

UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

MIN_BODY_CHARS = 20
MAX_TOTAL_CHARS = 50_000

SUSPICIOUS = "suspicious"
LEGITIMATE = "legitimate"
UNCERTAIN = "uncertain"


def validate_email(subject, body):
    subject = subject or ""
    body = body or ""
    problems = []

    if not body.strip():
        problems.append("The email body is empty. Paste the text of the email you want to check.")
    elif len(body.strip()) < MIN_BODY_CHARS:
        problems.append(
            f"The email body is very short (under {MIN_BODY_CHARS} characters). "
            "The model needs more text to give a useful result."
        )
    elif not any(ch.isalpha() for ch in body):
        problems.append("The email body contains no letters, so there is nothing for the model to read.")

    if len(subject) + len(body) > MAX_TOTAL_CHARS:
        problems.append(
            f"The email is too long (over {MAX_TOTAL_CHARS:,} characters). "
            "Paste a single email only."
        )
    return problems


def combine_text(subject, body):
    return (subject or "") + "\n" + (body or "")


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run 'python train.py' first."
        )
    return joblib.load(path)


def _positive_index(pipeline):
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError(f"Expected labels 0 and 1, but the model has {classes}.")
    return classes.index(1)


def recommendation_for(score):
    if score > UNCERTAIN_HIGH:
        return SUSPICIOUS
    if score < UNCERTAIN_LOW:
        return LEGITIMATE
    return UNCERTAIN


def classify(pipeline, subject, body):
    text = combine_text(subject, body)
    score = float(pipeline.predict_proba([text])[0, _positive_index(pipeline)])
    return {"score": score, "recommendation": recommendation_for(score)}


def explain(pipeline, subject, body, top_n=8):
    vectoriser = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["logreg"]
    text = combine_text(subject, body)

    x = vectoriser.transform([text])
    terms = vectoriser.get_feature_names_out()
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    sign = 1.0 if _positive_index(pipeline) == 1 else -1.0

    idx = x.indices
    values = x.data
    contributions = sign * values * coefs[idx]

    table = pd.DataFrame({
        "term": terms[idx],
        "tfidf": values,
        "coefficient": sign * coefs[idx],
        "contribution": contributions,
    })

    towards_suspicious = (
        table[table["contribution"] > 0]
        .sort_values("contribution", ascending=False).head(top_n)
        .reset_index(drop=True)
    )
    towards_legitimate = (
        table[table["contribution"] < 0]
        .sort_values("contribution", ascending=True).head(top_n)
        .reset_index(drop=True)
    )

    log_odds = sign * intercept + float(contributions.sum())
    model_log_odds = sign * float(pipeline.decision_function([text])[0])

    return {
        "towards_suspicious": towards_suspicious,
        "towards_legitimate": towards_legitimate,
        "n_known_terms": int(len(idx)),
        "intercept": sign * intercept,
        "log_odds": log_odds,
        "matches_model": bool(np.isclose(log_odds, model_log_odds)),
    }


def _quote_list(terms):
    quoted = [f"'{t}'" for t in terms]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def explanation_text(explanation, n=3):
    sus = list(explanation["towards_suspicious"]["term"].head(n))
    leg = list(explanation["towards_legitimate"]["term"].head(n))
    parts = []
    if sus:
        parts.append(f"The terms that pushed the score most towards suspicious were {_quote_list(sus)}.")
    if leg:
        parts.append(f"The terms that pushed it most towards legitimate were {_quote_list(leg)}.")
    if not parts:
        parts.append("None of the words in this email were in the model's vocabulary, "
                     "so the score comes from the model's baseline only.")
    return " ".join(parts)