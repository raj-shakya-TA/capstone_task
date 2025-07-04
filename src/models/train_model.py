import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from src.data.load_data import load_dataset

def main():
    with open("config/params.yaml") as f:
        config = yaml.safe_load(f)

    data = load_dataset(config["data"]["raw_path"])
    X = data.drop(columns=[config["data"]["target"]])
    y = data[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )

    p = config["train"]["p"]
    model = make_pipeline(PolynomialFeatures(p), LinearRegression())
    model.fit(X_train, y_train)
    print(f"Model R2 Score: {model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    main()
