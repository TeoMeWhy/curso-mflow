# %%
import mlflow.client
import pandas as pd

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# %%

client = mlflow.client.MlflowClient()
version = max([int(i.version) for i in client.get_latest_versions("churn-tmw")])

# %%
model = mlflow.sklearn.load_model(f"models:/churn-tmw/{version}")

# %%

df = pd.read_csv("data/abt.csv", sep=",")
df

# %%
X = df.head()[model.feature_names_in_]
proba = model.predict_proba(X)
proba

# %%
