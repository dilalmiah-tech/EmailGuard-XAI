import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_utils as mu
from train import build_pipeline


@pytest.fixture(scope="module")
def tiny_pipeline():
    suspicious = ["verify your account now click here", "urgent password expires login now",
                  "claim your prize now click here", "account suspended confirm your details"]
    legitimate = ["meeting notes attached for tuesday", "lunch tomorrow with the team",
                  "comments on the draft report", "project update and next steps for tuesday"]
    texts = [mu.combine_text("", t) for t in suspicious + legitimate] * 3
    labels = ([1] * 4 + [0] * 4) * 3
    pipeline = build_pipeline()
    pipeline.set_params(tfidf__min_df=1, tfidf__max_df=1.0)
    pipeline.fit(texts, labels)
    return pipeline


def test_empty_body_is_rejected():
    assert mu.validate_email("Hello", "   ")


def test_short_body_is_rejected():
    assert mu.validate_email("", "too short")


def test_body_without_letters_is_rejected():
    assert mu.validate_email("", "1234567890 !!!! 1234567890 ????")


def test_very_long_email_is_rejected():
    assert mu.validate_email("", "a" * (mu.MAX_TOTAL_CHARS + 1))


def test_normal_email_is_accepted():
    assert mu.validate_email("Meeting", "Hi all, the notes from Tuesday are attached.") == []


def test_missing_subject_is_allowed():
    assert mu.validate_email(None, "Hi all, the notes from Tuesday are attached.") == []


def test_combine_text_matches_training_format():
    assert mu.combine_text("Subject", "Body") == "Subject\nBody"
    assert mu.combine_text(None, "Body") == "\nBody"


@pytest.mark.parametrize("score, expected", [
    (0.10, mu.LEGITIMATE),
    (mu.UNCERTAIN_LOW, mu.UNCERTAIN),
    (0.50, mu.UNCERTAIN),
    (mu.UNCERTAIN_HIGH, mu.UNCERTAIN),
    (0.90, mu.SUSPICIOUS),
])
def test_uncertainty_band(score, expected):
    assert mu.recommendation_for(score) == expected


def test_classify_returns_score_between_0_and_1(tiny_pipeline):
    out = mu.classify(tiny_pipeline, "", "verify your account now click here")
    assert 0.0 <= out["score"] <= 1.0
    assert out["recommendation"] in {mu.SUSPICIOUS, mu.LEGITIMATE, mu.UNCERTAIN}


def test_contributions_add_up_to_model_output(tiny_pipeline):
    subject, body = "", "urgent: verify your account details for the tuesday meeting"
    exp = mu.explain(tiny_pipeline, subject, body, top_n=1000)
    model_log_odds = tiny_pipeline.decision_function([mu.combine_text(subject, body)])[0]
    total = (exp["intercept"]
             + exp["towards_suspicious"]["contribution"].sum()
             + exp["towards_legitimate"]["contribution"].sum())
    assert total == pytest.approx(model_log_odds)
    assert exp["matches_model"]


def test_explanation_directions_are_correct(tiny_pipeline):
    exp = mu.explain(tiny_pipeline, "", "verify your account for the tuesday meeting notes")
    assert (exp["towards_suspicious"]["contribution"] > 0).all()
    assert (exp["towards_legitimate"]["contribution"] < 0).all()


def test_unknown_words_give_no_contributions(tiny_pipeline):
    exp = mu.explain(tiny_pipeline, "", "zzqx wobble flarn quibbit snorkelwhip")
    assert exp["n_known_terms"] == 0
    assert "vocabulary" in mu.explanation_text(exp)


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python train.py"):
        mu.load_model(tmp_path / "missing.joblib")