import streamlit as st
import requests

#  PAGE CONFIG 
st.set_page_config(
    page_title="Bank Churn – Azure MLOps Demo",
    page_icon="🏦",
    layout="centered"
)

#  HEADER 
st.markdown(
    "<h1 style='text-align:center;'>🏦 Bank Churn Prediction</h1>",
    unsafe_allow_html=True
)

#  API CONFIG
BASE_URL = "https://app-churn-api.whiteflower-49131d93.francecentral.azurecontainerapps.io"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"

st.divider()

#  HEALTH CHECK 
st.subheader("🩺 API Health Check")

if st.button("Tester /health"):
    try:
        with st.spinner("Vérification de l'API..."):
            r = requests.get(HEALTH_URL, timeout=5)

        if r.status_code == 200:
            st.success("API en ligne – Modèle chargé")
            st.json(r.json())
        else:
            st.error("API indisponible")

    except requests.exceptions.RequestException:
        st.error("Impossible de joindre l'API")

st.divider()

#  FORM 
st.subheader("📊 Données client")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 90, 35)
        tenure = st.slider("Tenure", 0, 10, 5)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    with col2:
        balance = st.number_input("Balance", value=50000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_card = st.selectbox("Has Credit Card", [0, 1])
        active = st.selectbox("Active Member", [0, 1])
        salary = st.number_input("Estimated Salary", value=75000.0)

    submit = st.form_submit_button("🚀 Prédire le churn")

#  ONE-HOT ENCODING 
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

# PREDICTION 
if submit:
    try:
        with st.spinner("⏳ Calcul de la prédiction..."):
            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=10
            )

        if response.status_code == 200:
            result = response.json()
            churn = result.get("prediction", 0)
            prob = result.get("probability", 0)

            st.divider()
            st.subheader("📈 Résultat de la prédiction")

            if churn == 1:
                st.error(f" Client à risque de churn ({prob:.2%})")
                st.metric("Risque", "ÉLEVÉ", f"{prob:.2%}")
            else:
                st.success(f"Client fidèle ({1 - prob:.2%})")
                st.metric("Risque", "FAIBLE", f"{1 - prob:.2%}")

            st.progress(min(int(prob * 100), 100))

            with st.expander("🔍 Voir la réponse brute de l’API"):
                st.json(result)

        else:
            st.error("Erreur lors de l'appel API")
            st.write(response.text)

    except requests.exceptions.RequestException:
        st.error("Erreur réseau lors de l'appel API")

#  FOOTER 
st.divider()
st.caption("Projet MLOps – Déploiement Azure – CI/CD GitHub Actions")
