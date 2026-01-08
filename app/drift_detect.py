# app/drift_detect.py
import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = "data/bank_churn.csv"
PRODUCTION_PATH = "data/production_data.csv"

def detect_drift(threshold=0.05):
    ref = pd.read_csv(REFERENCE_PATH)
    prod = pd.read_csv(PRODUCTION_PATH)

    drifted_features = []

    numeric_cols = ref.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        stat, p_value = ks_2samp(ref[col], prod[col])

        if p_value < threshold:
            drifted_features.append({
                "feature": col,
                "p_value": round(p_value, 6)
            })

    return {
        "status": "success",
        "features_analyzed": len(numeric_cols),
        "features_drifted": len(drifted_features),
        "details": drifted_features
    }
