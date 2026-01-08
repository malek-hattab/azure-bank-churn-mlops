# drift_data_gen.py
import pandas as pd
import numpy as np

# Charger les données de référence
df = pd.read_csv("data/bank_churn.csv")

# Copier pour simuler la prod
prod_df = df.copy()

#  Simuler un DRIFT
prod_df["Age"] = prod_df["Age"] + np.random.normal(5, 3, size=len(prod_df))
prod_df["Balance"] = prod_df["Balance"] * np.random.uniform(1.2, 1.5)

# Sauvegarder les données "production"
prod_df.to_csv("data/production_data.csv", index=False)

print(" production_data.csv généré avec drift")
