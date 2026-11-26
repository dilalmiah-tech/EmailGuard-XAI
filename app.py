from datetime import datetime

import streamlit as st

import model_utils as mu

st.set_page_config(page_title="EmailGuard-XAI", page_icon="🛡️", layout="centered")

EXAMPLES = {
    "Choose an example...": None,
    "Account warning": (
        "Urgent: your account will be suspended",
        "Dear customer,\n\nWe detected unusual activity on your account. To avoid "
        "suspension you must verify your details within 24 hours using the link below:\n"
        "http://account-verify.example/login\n\nFailure to do so will result in "
        "permanent closure.\n\nSecurity Team",
    ),
    "Meeting notes": (
        "Notes from Tuesday's project meeting",
        "Hi all,\n\nThanks for coming on Tuesday. I've attached the notes from the "
        "meeting. Main actions: Sam will update the test plan, and I'll circulate the "
        "revised timeline by Friday. Let me know if I missed anything.\n\nBest,\nPriya",
    ),
    "Parcel delivery": (
        "Your parcel delivery",
        "Hello,\n\nWe tried to deliver your parcel today but nobody was home. You can "
        "choose a new delivery date on our website. Reference: 4821-77.\n\n"
        "Regards,\nDelivery Services",
    ),
}

STATE_TEXT = {
    mu.SUSPICIOUS: ("⚠️", "AI recommendation: Suspicious",
                    "The model's score is above the uncertain band."),
    mu.LEGITIMATE: ("✅", "AI recommendation: Likely legitimate",
                    "The model's score is below the uncertain band."),
    mu.UNCERTAIN: ("❔", "AI is uncertain: manual review recommended",
                   "The score is inside the provisional uncertain band "
                   f"({mu.UNCERTAIN_LOW}–{mu.UNCERTAIN_HIGH}), so the AI makes no recommendation."),
}

DECISION_LABELS = {"suspicious": "Suspicious", "legitimate": "Legitimate", "deferred": "Deferred"}


def init_state():
    defaults = {
        "subject": "",
        "body": "",
        "example": "Choose an example...",
        "result": None,
        "decision": None,
        "history": [],
        "email_counter": 0,
        "errors": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_result():
    st.session_state.result = None
    st.session_state.decision = None
    st.session_state.errors = []


def load_example():
    choice = EXAMPLES.get(st.session_state.example)
    if choice:
        st.session_state.subject, st.session_state.body = choice
    clear_result()


def clear_inputs():
    st.session_state.subject = ""
    st.session_state.body = ""
    st.session_state.example = "Choose an example..."
    clear_result()


def record_decision(user_label, decision_type):
    result = st.session_state.result
    entry = {
        "Email": f"#{result['email_id']}",
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Subject (preview)": (result["subject"][:40] + "…") if len(result["subject"]) > 40 else result["subject"],
        "AI recommendation": STATE_TEXT[result["recommendation"]][1].split(": ")[-1],
        "Model score": round(result["score"], 2),
        "Your decision": DECISION_LABELS[user_label],
        "Decision type": decision_type,
        "_email_id": result["email_id"],
    }
    st.session_state.history.append(entry)
    st.session_state.decision = entry


def undo_last_decision():
    if not st.session_state.history:
        return
    removed = st.session_state.history.pop()
    result = st.session_state.result
    if result and removed["_email_id"] == result["email_id"]:
        st.session_state.decision = None


def clear_history():
    st.session_state.history = []
    st.session_state.decision = None


@st.cache_resource
def get_model():
    return mu.load_model()


def show_assessment_and_decision(result):
    st.divider()
    st.subheader(f"2. AI assessment (email #{result['email_id']})")

    icon, headline, detail = STATE_TEXT[result["recommendation"]]
    box = {mu.SUSPICIOUS: st.warning, mu.LEGITIMATE: st.success, mu.UNCERTAIN: st.info}[result["recommendation"]]
    box(f"**{headline}**\n\n{detail}", icon=icon)

    score = result["score"]
    st.metric("Model score", f"{score:.2f}")
    st.progress(score, text="0 = text looks legitimate · 1 = text looks suspicious")
    st.caption(
        "The model score shows how strongly the text matches suspicious emails in the "
        "training data. It is **not** a calibrated probability: 0.90 does not mean a 90% "
        f"chance of phishing. The uncertain band ({mu.UNCERTAIN_LOW}–{mu.UNCERTAIN_HIGH}) "
        "is provisional and has not been validated."
    )

    exp = result["explanation"]
    st.markdown("#### Why the AI gave this score")
    st.markdown(mu.explanation_text(exp))

    largest = max(
        [abs(v) for v in exp["towards_suspicious"]["contribution"]]
        + [abs(v) for v in exp["towards_legitimate"]["contribution"]]
        + [1e-9]
    )


    def contribution_table(df, column):
        if df.empty:
            column.caption("No terms in this direction.")
            return
        column.dataframe(
            df.assign(strength=df["contribution"].abs())[["term", "strength"]],
            hide_index=True,
            column_config={
                "term": st.column_config.TextColumn("Term"),
                "strength": st.column_config.ProgressColumn(
                    "Contribution", min_value=0.0, max_value=float(largest), format="%.3f"
                ),
            },
        )


    left, right = st.columns(2)
    left.markdown("**⬆️ Towards suspicious**")
    contribution_table(exp["towards_suspicious"], left)
    right.markdown("**⬇️ Towards legitimate**")
    contribution_table(exp["towards_legitimate"], right)

    with st.expander("How is this explanation calculated?"):
        st.markdown(
            "The model adds up a weight for every word or two-word phrase it recognises "
            "in this email. Each term's contribution = how prominent the term is in this "
            "email (its TF-IDF value) × the weight the model learned for it. These are the "
            "model's real numbers, not a separate guess about its reasoning.\n\n"
            f"- Terms the model recognised in this email: {exp['n_known_terms']}\n"
            f"- Baseline (intercept): {exp['intercept']:.3f}\n"
            f"- Baseline + all contributions: {exp['log_odds']:.3f} "
            f"({'matches' if exp['matches_model'] else 'does NOT match'} the model's own output)\n\n"
            "Words the model never saw in training count for nothing, and a high-weight "
            "word can reflect quirks of the 2008 dataset rather than real phishing signs."
        )

    st.markdown("#### Check before you decide")
    st.markdown(
        "The AI can't see these, so check them yourself:\n"
        "- Do you recognise the sender's address, and does it match who they claim to be?\n"
        "- If you hover over a link, does the real address match the text?\n"
        "- Does it pressure you to act quickly, or ask for passwords, payment or personal details?\n"
        "- Were you expecting this email?"
    )

    st.divider()
    st.subheader("3. Your decision")

    rec = result["recommendation"]
    if rec == mu.SUSPICIOUS:
        options = [("Accept: mark as suspicious", "suspicious", "Accepted AI"),
                   ("Override: mark as legitimate", "legitimate", "Overrode AI")]
    elif rec == mu.LEGITIMATE:
        options = [("Accept: mark as legitimate", "legitimate", "Accepted AI"),
                   ("Override: mark as suspicious", "suspicious", "Overrode AI")]
    else:
        options = [("Mark as suspicious", "suspicious", "Own decision (AI uncertain)"),
                   ("Mark as legitimate", "legitimate", "Own decision (AI uncertain)")]
    options.append(("Defer: decide later", "deferred", "Deferred"))

    decided = st.session_state.decision is not None
    cols = st.columns(len(options))
    for col, (label, user_label, decision_type) in zip(cols, options):
        col.button(label, on_click=record_decision, args=(user_label, decision_type),
                   disabled=decided, key=f"decide_{user_label}")

    if decided:
        d = st.session_state.decision
        st.success(f"**Recorded:** you marked email {d['Email']} as **{d['Your decision'].lower()}** "
                   f"({d['Decision type'].lower()}). Undo it if you change your mind.", icon="📝")
        st.button("↩️ Undo this decision", on_click=undo_last_decision)
    else:
        st.caption("Recording a decision only adds it to the history below. It does nothing to your inbox.")


init_state()

try:
    pipeline = get_model()
except FileNotFoundError as err:
    st.error(f"**The app cannot run yet.** {err}", icon="🛑")
    st.stop()
except Exception as err:
    st.error(f"**The model could not be loaded.** {err}\n\nTry running `python train.py` again.", icon="🛑")
    st.stop()

with st.sidebar:
    st.header("About EmailGuard-XAI")
    st.markdown(
        "An **advisory** tool. It suggests whether an email looks suspicious and "
        "shows which words influenced that suggestion. **You make the final decision.**"
    )
    st.markdown("**AI assistance:** 🟢 Active (advice only)")
    st.subheader("What it can't do")
    st.markdown(
        "- It reads only the subject and body text. It does **not** check the "
        "sender's address, where links really go, or attachments.\n"
        "- It was trained on the CEAS 2008 dataset, which mixes spam and phishing "
        "and is now old. It flags *spam-like or phishing-like* text, and can be wrong "
        "about modern phishing.\n"
        "- The **model score** is not a probability and not a guarantee."
    )
    st.subheader("Privacy")
    st.markdown(
        "Nothing is sent to the internet or saved to disk. The session history keeps "
        "a short subject preview, never the email body, and disappears when you "
        "close this tab."
    )

st.title("🛡️ EmailGuard-XAI")
st.caption("AI-assisted checking for suspicious emails. The AI advises; you decide. "
           "Nothing here deletes, blocks or reports any email.")

st.subheader("1. Email")
st.selectbox("Try an example email (optional)", list(EXAMPLES), key="example", on_change=load_example)
st.text_input("Subject", key="subject", on_change=clear_result,
              placeholder="Paste the subject line (optional)")
st.text_area("Email body", key="body", on_change=clear_result, height=200,
             placeholder="Paste the text of the email here")

col_a, col_b = st.columns([1, 1])
analyse_clicked = col_a.button("Analyse email", type="primary")
col_b.button("Clear", on_click=clear_inputs)

if analyse_clicked:
    clear_result()
    problems = mu.validate_email(st.session_state.subject, st.session_state.body)
    if problems:
        st.session_state.errors = problems
    else:
        try:
            outcome = mu.classify(pipeline, st.session_state.subject, st.session_state.body)
            explanation = mu.explain(pipeline, st.session_state.subject, st.session_state.body)
        except Exception as err:
            st.session_state.errors = [f"The analysis failed: {err}"]
        else:
            st.session_state.email_counter += 1
            st.session_state.result = {
                **outcome,
                "explanation": explanation,
                "email_id": st.session_state.email_counter,
                "subject": st.session_state.subject,
                "body": st.session_state.body,
            }

for problem in st.session_state.errors:
    st.error(problem, icon="🛑")

result = st.session_state.result
if result and (result["subject"] != st.session_state.subject or result["body"] != st.session_state.body):
    clear_result()
    result = None

if result is None:
    if not st.session_state.errors:
        st.info("Paste an email and select **Analyse email** to get the AI's assessment.", icon="ℹ️")
else:
    show_assessment_and_decision(result)

st.divider()
st.subheader("4. Session history")
history = st.session_state.history
if not history:
    st.caption("No decisions recorded yet.")
else:
    st.dataframe([{k: v for k, v in h.items() if not k.startswith("_")} for h in history],
                 hide_index=True)
    types = [h["Decision type"] for h in history]
    st.caption(f"Accepted AI: {types.count('Accepted AI')} · Overrode AI: {types.count('Overrode AI')} · "
               f"Own decision: {types.count('Own decision (AI uncertain)')} · Deferred: {types.count('Deferred')}")
    h1, h2 = st.columns(2)
    h1.button("↩️ Undo last decision", on_click=undo_last_decision, key="undo_history")
    h2.button("Clear history", on_click=clear_history)