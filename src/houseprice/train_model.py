import pandas as pd
import joblib
import yaml
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_config(path="config/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    df = pd.read_csv(config["data"]["processed_data_path"])

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_params = config["model"]["params"]
    model = LinearRegression(**model_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    os.makedirs(os.path.dirname(config["output"]["model_path"]), exist_ok=True)
    joblib.dump(model, config["output"]["model_path"])

    with open(config["output"]["metrics_path"], "w") as f:
        f.write(f"Mean Squared Error: {mse:.2f}\n")

if __name__ == "__main__":
    main()
