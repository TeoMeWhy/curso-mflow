# %%
import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble

# %%

import mlflow
mlflow.set_tracking_uri("http://localhost:5000/")
mlflow.set_experiment(experiment_id=566318506401660250)

# %%
df = pd.read_csv("data/abt.csv", sep=",")

features = df.columns[2:-1]
target = "flag_churn"

X = df[features]
y = df[target]

df.head()
# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.2,
                                                                    random_state=42)

print("Taxa de resposta train:", y_train.mean())
print("Taxa de resposta test:", y_test.mean())

# %%

with mlflow.start_run():

    mlflow.sklearn.autolog()

    # clf = tree.DecisionTreeClassifier(min_samples_leaf=100,
    #                                   min_samples_split=10,
    #                                   random_state=42)

    clf = ensemble.RandomForestClassifier(n_estimators=500,
                                          min_samples_leaf=35,
                                          random_state=42)
    clf.fit(X_train, y_train)

    y_train_predict = clf.predict(X_train)
    y_test_predict = clf.predict(X_test)

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    acc_test = metrics.accuracy_score(y_test, y_test_predict)

    mlflow.log_metrics({"acc_train": acc_train, "acc_test": acc_test})


# %%
print("Acurácia train:", acc_train)
print("Acurácia test:", acc_test)
# %%
