import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.models.signature import ModelSignature

from mlflow.types.schema import Schema, ColSpec

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def get_Schema():

    input_schema = Schema([
    ColSpec("float", "fixed acidity"),
    ColSpec("double", "volatile"),
    ColSpec("double", "citric acid"),
    ColSpec("float", "residual sugar"),
    ColSpec("integer", "total sulfur dioxide	"),
    ColSpec("long", "free sulfur dioxide	"),
    ColSpec("double", "chlorides"),
    ColSpec("double", "density"),
    ColSpec("double", "Ph"),
    ColSpec("double", "alcohol"),
    ColSpec("double", "sulphates"),
    ])
    output_schema = Schema([ColSpec("integer","quality")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    input_example = {
        "sulphates": 5.1,
        "alcohol": 3.5,
        "Ph": 1.4,
        "density": 0.2
    }

    return signature,input_example



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        signature,input_ex = get_Schema()
        mlflow.sklearn.log_model(lr, "model",signature=signature,input_example = data.loc[:, data.columns != 'quality'].head(1))