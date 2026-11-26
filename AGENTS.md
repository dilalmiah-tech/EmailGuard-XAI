# AGENTS.md

Guidance for AI coding assistants working on EmailGuard-XAI. Read this file and `README.md` before making changes.

# Purpose

EmailGuard-XAI is an advisory tool that classifies pasted email text as suspicious, likely legitimate or uncertain, and explains the result using the model's own weights. The user always makes the final decision. It is a university project for the Intelligent Interactive Systems module and is being finalised for submission.

# Architecture

| File | Role |
|---|---|
| `train.py` | Loads `data/CEAS_08.csv`, removes duplicates, makes an 80/20 stratified split (seed 42), trains a TF-IDF + logistic regression pipeline, and writes `reports/metrics.txt`, `reports/confusion_matrix.png` and `models/emailguard_pipeline.joblib` |
| `model_utils.py` | Input validation, text formatting, classification, the uncertainty band (0.35–0.65) and explanations. No Streamlit code |
| `app.py` | Streamlit interface: input, assessment, decision controls, undo and session history |
| `tests/test_model_utils.py` | pytest tests for `model_utils.py` |

# Technology stack

Python 3.12, pandas, scikit-learn, joblib, matplotlib, Streamlit and pytest. Do not add dependencies, external APIs, databases or network calls.

# Commands (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
python train.py          # train and evaluate (needs data/CEAS_08.csv)
python -m pytest         # run automated tests
streamlit run app.py     # launch the app at http://localhost:8501
```

# Coding conventions

- Keep the three-file structure. Model logic belongs in `model_utils.py`, not `app.py`.
- Email text must be built as `subject + "\n" + body` in both training and the app.
- Fit the vectoriser on training data only, through the scikit-learn pipeline.
- Use `pathlib` paths relative to the project folder.
- Write code that a student with intermediate Python knowledge can read and change.
- Use British English in user-facing text.
- Run `python -m pytest` after any change. All tests must pass.
- The project is being finalised: do not add features or restructure code unless the student asks.

# Human control and explainability

- The AI gives advice only. Never add automatic deletion, blocking, reporting or any other action on emails.
- Keep accept and override available when the AI makes a recommendation. Allow independent decisions when uncertain. Keep defer and undo available where applicable.
- In the uncertain band, recommend manual review and make no recommendation.
- Explanations must come from the real model: TF-IDF value × coefficient. Never add generated or invented explanations.
- Call the score a "model score". Never present it as a calibrated probability or a guaranteed risk level.
- Do not rely on colour alone. Every state needs text and an icon.
- Clear any previous result when the input changes.
- Keep the warnings about limitations and overreliance.

# Privacy and security

- Do not store email bodies. History stays in Streamlit session state only, with a short subject preview.
- Do not write user input to disk, logs or external services.
- Never commit `data/CEAS_08.csv`, `models/`, `.venv/`, API keys or credentials.

# Honesty requirements

- Never invent evaluation results, test results, participant data or user-study findings. Report only figures from `reports/metrics.txt` or results the student has confirmed.
- Never describe planned features as implemented. Planned: turning AI suggestions off after repeated overrides, evaluation conditions, a calibration check, a newer dataset and an accessibility review.
- Do not claim that the system reduces overreliance or detects real-world phishing reliably. Describe it as a suspicious-email classifier supporting phishing-risk assessment.
- Label 1 is treated as suspicious, but this has not been independently verified. Keep that caveat.
- Do not fabricate a development history or commit messages.