import pandas as pd
import joblib
import yaml

# Load config
with open("config.yml") as f:
    config = yaml.safe_load(f)

# Load test data
df = pd.read_csv(config["data_source"]["local_path"])
X_test = df.drop("SalePrice", axis=1)

# Load model
model = joblib.load("artifacts/model.joblib")

# Predict
predictions = model.predict(X_test)
print("Predictions:", predictions[:5])
