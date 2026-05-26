import streamlit as st
import joblib
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Loan Application",
                   page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("model_prod_files.pkl")
    return artifacts["model"], artifacts["cat_encod"], artifacts["num_encod"]


model, cat_encod, num_encod = load_artifacts()

NUM_COLS = ["age", "balance", "day", "duration", "campaign", "pdays"]
CAT_COLS = ["job", "marital", "education", "contact", "month", "poutcome"]

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .block-container { padding-top: 2rem; }
    .stButton>button {
        background-color: #1a3c5e;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton>button:hover { background-color: #25527e; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏦 Loan Application Form")
st.markdown(
    "Please enter the following details for verification. All information is kept confidential.")
st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("loan_form"):

    st.markdown("#### 👤 Personal Details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=95, value=30)
        marital = st.selectbox("Marital Status",  [
                               "single", "married", "divorced"])
        education = st.selectbox(
            "Education Level", ["primary", "secondary", "tertiary"])
    with c2:
        job = st.selectbox("Occupation", ["admin", "management", "retired",
                                          "services", "student", "technician"])
        balance = st.number_input("Account Balance (€)", value=1000, step=100)

    st.markdown("#### 💳 Financial Details")
    c3, c4 = st.columns(2)
    with c3:
        default_val = st.selectbox(
            "Any existing credit default?", ["no", "yes"])
        housing = st.selectbox("Do you have a housing loan?",  ["no", "yes"])
    with c4:
        personal = st.selectbox("Do you have a personal loan?", ["no", "yes"])
        previous = st.number_input(
            "Number of previous contacts", min_value=0, value=0)

    st.markdown("#### 📞 Contact Details")
    c5, c6 = st.columns(2)
    with c5:
        contact = st.selectbox("Preferred contact type",
                               ["cellular", "telephone"])
        month = st.selectbox("Month of last contact",
                             ["jan", "feb", "mar", "apr", "may", "jun",
                                 "jul", "aug", "sep", "oct", "nov", "dec"])
    with c6:
        day = st.number_input("Day of last contact",
                              min_value=1, max_value=31, value=15)
        duration = st.number_input(
            "Last call duration (seconds)", min_value=0, value=180)
        pdays = st.number_input(
            "Days since last contacted (-1 if never)", min_value=-1, value=-1)
        poutcome = st.selectbox("Outcome of previous contact", [
                                "unknown", "failure", "success"])

    st.markdown("")
    submitted = st.form_submit_button(
        "Submit Application", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    campaign = 1  # hidden from UI, fixed internally

    with st.spinner("🔍 Verifying your request..."):
        time.sleep(2)

    # 1. Scale numerical columns
    num_df = pd.DataFrame(
        [[age, balance, day, duration, campaign, pdays]], columns=NUM_COLS)
    num_scaled = pd.DataFrame(num_encod.transform(num_df), columns=NUM_COLS)

    # 2. Encode categorical columns (only valid trained categories used)
    known = {col: list(cats)
             for col, cats in zip(CAT_COLS, cat_encod.categories_)}

    def safe(col, val):
        return val if val in known[col] else known[col][0]

    cat_df_raw = pd.DataFrame([[
        safe("job",       job),
        safe("marital",   marital),
        safe("education", education),
        safe("contact",   contact),
        safe("month",     month),
        safe("poutcome",  poutcome),
    ]], columns=CAT_COLS)

    cat_encoded = cat_encod.transform(cat_df_raw)
    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()
    cat_df = pd.DataFrame(
        cat_encoded, columns=cat_encod.get_feature_names_out(CAT_COLS))

    # 3. Raw binary + previous
    extra_df = pd.DataFrame([[
        1 if default_val == "yes" else 0,
        1 if housing == "yes" else 0,
        1 if personal == "yes" else 0,
        previous
    ]], columns=["default", "housing_loan", "personal_loan", "previous"])

    # 4. Assemble in exact model column order
    full_df = pd.concat([num_scaled, cat_df, extra_df], axis=1)
    expected = list(model.feature_names_in_)
    for col in expected:
        if col not in full_df.columns:
            full_df[col] = 0
    full_df = full_df[expected]

    # 5. Predict
    pred = model.predict(full_df)[0]
    proba = model.predict_proba(full_df)[0]
    approved = int(pred) == 1

    # 6. Show result
    st.divider()
    if approved:
        st.success("## ✅ Congratulations! Your loan has been **Approved**.")
        st.markdown(
            "Based on the details provided, your application has been accepted. Our team will contact you shortly.")
    else:
        st.error("## ❌ We're sorry. Your loan has been **Rejected**.")
        st.markdown(
            "Based on the details provided, we are unable to approve your application at this time.")
