# %%

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

model = mlflow.sklearn.load_model("models:/churn-tmw/2")

# %%

model.feature_names_in_

# %%

import pandas as pd
df = pd.read_csv("data/abt.csv", sep=",")
df

# %%

X = df.head()[model.feature_names_in_]

# %%

proba = model.predict_proba(X)
proba
# %%
