import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide")

# ---------- Style ----------
st.markdown("""
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }

/* Header */
.app-title { font-size: 2rem; font-weight: 800; color: #1E293B; margin: 0; }
.app-sub { color: #64748B; margin: 0.2rem 0 0.8rem; }
.app-divider { border: none; border-top: 1px solid #E2E8F0; margin: 0.4rem 0 1.2rem; }

/* Cards that hold the inputs */
[class*="st-key-card"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.2rem 0.6rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.section-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.4rem;
}
[data-testid="stWidgetLabel"] p { color: #64748B; font-weight: 600; }
.total-line { color: #64748B; font-size: 0.9rem; margin-bottom: 0.6rem; }
.total-line b { color: #0F172A; }

/* Button */
.stButton>button {
    width: 100%;
    background: #1E293B;
    color: #FFFFFF;
    font-weight: 700;
    padding: 0.7rem;
    border-radius: 12px;
    border: none;
}
.stButton>button:hover { background: #0F172A; color: #FFFFFF; }

/* Result card */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.6rem;
    margin: 1rem 0 0.8rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.badges { display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
.badge { padding: 0.3rem 0.85rem; border-radius: 999px; font-size: 0.85rem; font-weight: 700; border: 1px solid; }
.badge-high      { background: #FEE2E2; color: #B91C1C; border-color: #FCA5A5; }
.badge-medium    { background: #FFEDD5; color: #C2410C; border-color: #FDBA74; }
.badge-low       { background: #D1FAE5; color: #047857; border-color: #6EE7B7; }
.badge-threshold { background: #FEF3C7; color: #D97706; border-color: #F59E0B; }
.result-proba { font-size: 3.6rem; font-weight: 800; line-height: 1.1; color: #0F172A; }
.result-sub { color: #64748B; }
.bar { position: relative; height: 12px; background: #E2E8F0; border-radius: 8px; margin: 1.2rem auto 0.5rem; max-width: 520px; }
.bar-fill { height: 100%; border-radius: 8px; }
.fill-high { background: #EF4444; }
.fill-medium { background: #F97316; }
.fill-low { background: #10B981; }
.bar-mark { position: absolute; top: -5px; width: 3px; height: 22px; background: #D97706; border-radius: 2px; }
.bar-legend { font-size: 0.85rem; color: #64748B; }

/* Recommendation */
.reco {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    color: #1E3A8A;
}

/* Sidebar numbers */
[data-testid="stMetricValue"] { color: #0F172A; font-weight: 800; }
[data-testid="stMetricLabel"] p { color: #64748B; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    saved = joblib.load("models/churn_model.joblib")   # pipeline + seuil, sauvegardés dans le notebook
    return saved["model"], saved["threshold"]


try:
    model, threshold = load_model()
except FileNotFoundError:
    st.error("Model file not found: models/churn_model.joblib. Run the modeling notebook first.")
    st.stop()

# ---------- Header ----------
st.markdown("""
<div class="app-title">Customer Churn Prediction</div>
<div class="app-sub">Estimate the probability that a telecom customer will leave.</div>
<hr class="app-divider">
""", unsafe_allow_html=True)

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    with st.container(key="card_profile"):
        st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with st.container(key="card_phone"):
        st.markdown('<div class="section-title">Phone and internet</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        # Les services dépendent du téléphone / d'internet : on évite les combinaisons impossibles
        if phone_service == "No":
            multiple_lines = "No phone service"
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

        def service(label):
            if internet_service == "No":
                return "No internet service"
            return st.selectbox(label, ["No", "Yes"])

        online_security = service("Online Security")
        online_backup = service("Online Backup")

with col2:
    with st.container(key="card_addons"):
        st.markdown('<div class="section-title">Add-on services</div>', unsafe_allow_html=True)
        device_protection = service("Device Protection")
        tech_support = service("Tech Support")
        streaming_tv = service("Streaming TV")
        streaming_movies = service("Streaming Movies")

    with st.container(key="card_account"):
        st.markdown('<div class="section-title">Account</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check",
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, 5.0)
        total_charges = tenure * monthly_charges   # estimé : ancienneté x facture mensuelle
        st.markdown(
            f'<div class="total-line">Total charges (estimated: tenure x monthly): <b>${total_charges:,.0f}</b></div>',
            unsafe_allow_html=True)

st.markdown("")

# ---------- Prediction ----------
if st.button("Predict churn"):
    input_data = {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_df = pd.DataFrame([input_data])

    # Le pipeline encode et met à l'échelle tout seul : pas de transformation à la main
    proba = model.predict_proba(input_df)[0, 1]   # probabilité de churn
    at_risk = proba >= threshold                  # seuil choisi dans le notebook

    if proba >= 0.7:
        risk, lvl = "High", "high"
    elif proba >= threshold:
        risk, lvl = "Medium", "medium"
    else:
        risk, lvl = "Low", "low"

    label = "AT RISK OF CHURN" if at_risk else "LOW RISK"

    if at_risk:
        reco = "<b>Recommendation:</b> offer a retention incentive, propose a longer contract, improve support."
    else:
        reco = "<b>Recommendation:</b> keep the service quality and the relationship."

    st.markdown(f"""
<div class="result-card">
<div class="badges">
<span class="badge badge-{lvl}">{label}</span>
<span class="badge badge-threshold">Decision threshold: {threshold}</span>
</div>
<div class="result-proba">{proba * 100:.1f}%</div>
<div class="result-sub">churn probability | risk level: {risk}</div>
<div class="bar">
<div class="bar-fill fill-{lvl}" style="width: {proba * 100:.1f}%"></div>
<div class="bar-mark" style="left: {threshold * 100:.0f}%"></div>
</div>
<div class="bar-legend">Amber marker = decision threshold</div>
</div>
<div class="reco">{reco}</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model info")
    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC", "0.841")
    m2.metric("Recall", "0.866")
    m3, m4 = st.columns(2)
    m3.metric("Precision", "0.463")
    m4.metric("Threshold", f"{threshold}")
    st.caption("Logistic regression, balanced class weights, 7,043 customers (80/20 split). Metrics on the test set.")

    st.markdown("---")
    st.markdown("**:red[What raises the risk]**")
    st.markdown("""
- Fiber optic internet
- Month-to-month contract
- Short tenure
""")
    st.markdown("**:green[What lowers it]**")
    st.markdown("""
- DSL internet
- Two-year contract
- Long tenure
""")

    st.markdown("---")
    st.caption("Built with Streamlit and scikit-learn. Learning project, not for real decisions.")