# EmailGuard-XAI


*Repository:* https://github.com/dilalmiah-tech/EmailGuard-XAI

**Submitted version:** `v1.0-submission`

EmailGuard-XAI is an explainable, AI-assisted tool for checking whether an email looks suspicious. It gives a recommendation, a model score and an explanation, and the user makes the final decision. It never deletes emails, blocks senders or takes security actions.

# Project status

| Feature | Status |
|---|---|
| TF-IDF + logistic regression classifier trained on CEAS_08 | Implemented |
| Input validation (empty, short, no letters, too long) | Implemented |
| Three states: suspicious (>0.65), likely legitimate (<0.35), uncertain (0.35–0.65, manual review) | Implemented |
| Explanation: TF-IDF value × coefficient per term, as tables and text | Implemented |
| Accept, override and defer controls, with undo | Implemented |
| Session-only history (no email body stored) | Implemented |
| Stale results cleared when the input changes | Implemented |
| Limitation and overreliance warnings | Implemented |
| Automated tests for `model_utils.py` | Implemented |
| Turning AI suggestions off after repeated overrides | Planned |
| Evaluation conditions (human only / with AI / with AI and explanation) | Planned |
| User study of trust or reliance | Not conducted |

States use icons and text, not colour alone. The score is labelled as uncalibrated.

# Technology stack

Python 3.12, pandas (data cleaning), scikit-learn (TF-IDF, logistic regression, split and metrics), joblib (saving the model), matplotlib (confusion matrix), Streamlit (interface and session state), pytest (tests). No paid APIs, external services or databases.


EmailGuard-XAI/
├── train.py           # data cleaning, training, evaluation
├── model_utils.py     # validation, classification, uncertainty, explanations
├── app.py             # Streamlit interface
├── tests/test_model_utils.py
├── reports/           # metrics.txt, confusion_matrix.png
├── data/              # CEAS_08.csv (not committed)
├── models/            # trained pipeline (not committed)
├── requirements.txt, .gitignore, README.md, AGENTS.md
```

# Dataset

`CEAS_08.csv` from the Phishing Email Curated Datasets on Zenodo (Champa, Rabbi and Zibran, 2024): https://zenodo.org/records/8339691. It is not included in the repository.

It has 39,154 rows (class 1: 21,842; class 0: 17,312). The model uses subject and body only. Removing duplicates before the 80/20 stratified split (seed 42) dropped 191 rows; none had conflicting labels.

Label 1 is treated as suspicious. This is unverified, and as CEAS 2008 began as a spam corpus, class 1 probably includes spam.

# Installation and running

powershell
git clone https://github.com/dilalmiah-tech/EmailGuard-XAI.git
cd EmailGuard-XAI
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
python -m pytest
streamlit run app.py
```

Put `CEAS_08.csv` in `data/` before training. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

# Evaluation and testing

Results from `train.py` on the test set (7,793 of 38,963 cleaned emails; positive class = suspicious):

| Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| 99.49% | 99.58% | 99.49% | 99.54% | 3,440 | 18 | 22 | 4,313 |

56 test emails (0.72%) fell in the uncertain band. Strong terms such as `your`, `replica` and `watches` (suspicious) and `opensuse` and `python` (legitimate) suggest spam and mailing-list artefacts, so the system is described as a suspicious-email classifier, not a validated phishing detector.

*Automated tests:* all 17 passed in 6.35 seconds. They cover validation, uncertainty boundaries, explanation contributions summing to the model output, and the missing-model error.

*Manual tests (Windows, local Streamlit):*

- The app launched.
- The three fictional examples scored as follows: Account warning 0.63 (uncertain), Meeting notes 0.10 (likely legitimate), Parcel delivery 0.72 (suspicious).
- Accept, override, defer and undo worked.
- History and counts updated correctly.
- Empty and short inputs gave error messages.
- Editing an email cleared the old assessment.
- The explanation check matched the model output for the account warning.

# Limitations and future work

- Senders, link destinations and attachments are not checked.
- The dataset is old and mixes spam with phishing; the obvious-looking account warning scored lower than the parcel message.
- The uncertainty band was set by hand.
- No user study was run, and no claim is made that the system reduces overreliance.
- Accessibility has not been tested with assistive technology.
- Planned: the assistance-off option, evaluation conditions, a calibration check, a newer phishing-specific dataset, and an accessibility review.

# Development record

1. *Scope:* I simplified Claude's initial design (SQLite logging, hashed identifiers, multi-step assistance policy) to three Python files and session state because of the deadline.
2. *Data:* I chose CEAS_08, inspected it, and set the preprocessing rules above.
3. *Training:* Claude generated `train.py` to my specification, and I ran it on the real data.
4. *Model review:* after I reviewed the top coefficients, I narrowed the claims made about the model.
5. *Application:* Claude generated `model_utils.py`, `app.py` and the tests to my requirements. I ran and checked them.

Commit links: Commit links:

- [Initial repository setup](https://github.com/dilalmiah-tech/EmailGuard-XAI/commit/2ce2315)
- [Classification model and evaluation pipeline](https://github.com/dilalmiah-tech/EmailGuard-XAI/commit/a400317)
- [Interactive application, explainability, tests and documentation](https://github.com/dilalmiah-tech/EmailGuard-XAI/commit/ad9d14d)

# AI assistance

- *Claude (Anthropic):* dataset options and plan; aided in generating Python code and compiling tests; drafted this README.
- *ChatGPT (OpenAI):* development guidance, code review and testing support.

*My role:*
- I made the design and scope decisions and the preprocessing rules.
- I ran training, the automated tests and the manual checks myself, and recorded the actual results.
- I checked the metrics against the confusion matrix and reviewed this README.
- Worked on the report.


# References

Champa, A.I., Rabbi, M.F. and Zibran, M.F. (2024) 'Curated datasets and feature analysis for phishing email detection with machine learning', *3rd IEEE International Conference on Computing and Machine Intelligence (ICMI)*, pp. 1–7. doi: 10.1109/ICMI60790.2024.10585821.

Champa, A.I., Rabbi, M.F. and Zibran, M.F. (2024) 'Why phishing emails escape detection: a closer look at the failure points', *12th International Symposium on Digital Forensics and Security (ISDFS)*, pp. 1–6.