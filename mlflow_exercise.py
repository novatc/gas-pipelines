import mlflow

import numpy as np

from sklearn.linear_model import LinearRegression

with mlflow.start_run():
    X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    y = np.array([1, 2, 3, 4, 5])
    model = LinearRegression()
    model.fit(X, y)
    model_info = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")


sklearn_pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
data = np.array([6, 7, 8, 9, 10]).reshape((-1, 1))

predictions = sklearn_pyfunc_model.predict(data)

# q: compare machine learning

