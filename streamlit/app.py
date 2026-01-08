import streamlit as st
import requests

# ================= CONFIG =================
st.set_page_config(
    page_title="Bank Churn – MLOps Demo",
    page_icon="🏦",
    layout="centered"
)

API_BASE_URL = "https://app-churn-api.whiteflower-49131d93.francecentral.azurecontainerapps.io"

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 🏦 Bank Churn MLOps")
    st.markdown("""
    **Architecture**
    - Streamlit (UI)
    - Azure Container Apps
    - FastAPI
    - ML model (Scikit-learn)
    
    **Fonctionnalités**
    - Health check API
    - Prédiction churn
    - Déploiement CI/CD
    - Data Drift (bonus)
    """)
    st.markdown("---")
    st.caption("Projet MLOps – Azure")

# ================= HEADER =================
st.markdown(
    """
    <h1 style="text-align:center;">🏦 Bank Churn Prediction</h1>
    <p style="text-align:center; color:gray;">
    Streamlit ➜ Azure Container Apps ➜ FastAPI
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ================= HEALTH CHECK =================
st.subheader("🩺 API Health Check")

if st.button("Tester l’API"):
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("API opérationnelle – Modèle chargé")
            st.json(response.json())
        else:
            st.error("API indisponible")
    except Exception as e:
        st.error(f"Erreur : {e}")

st.divider()

# ================= FORM =================
st.subheader("📊 Données client")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 90, 35)
        tenure = st.slider("Tenure (années)", 0, 10, 5)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    with col2:
        balance = st.number_input("Balance", value=50000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_card = st.selectbox("Has Credit Card", [0, 1])
        active = st.selectbox("Active Member", [0, 1])
        salary = st.number_input("Estimated Salary", value=75000.0)

    submitted = st.form_submit_button("🚀 Prédire le churn")

# ================= PAYLOAD =================
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

payload = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain
}

# ================= RESULT =================
if submitted:
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            proba = result["churn_probability"]
            prediction = result["prediction"]
            risk = result["risk_level"]

            st.divider()
            st.subheader("📈 Résultat")

            if prediction == 1:
                st.error(f"⚠️ Client à risque de churn ({proba:.2%})")
            else:
                st.success(f"✅ Client fidèle ({1 - proba:.2%})")

            st.progress(int(proba * 100))
            st.info(f"🔎 Niveau de risque : **{risk}**")

            with st.expander("🔍 Détails techniques"):
                st.json(result)

        else:
            st.error("Erreur API")
            st.text(response.text)

    except Exception as e:
        st.error(f"Erreur : {e}")

st.divider()
st.caption("© Projet MLOps – Azure – CI/CD GitHub Actions")
