from pathlib import Path
import subprocess
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = Path("data/queue_data.csv")
MODEL_PATH = Path("models/queue_model.joblib")

if not DATA_PATH.exists():
    subprocess.run([sys.executable, "generate_data.py"], check=True)

df = pd.read_csv(DATA_PATH)
features = [
    "people_waiting", "active_counters", "avg_service_time",
    "hour", "day_of_week", "service_type"
]
X = df[features]
y = df["waiting_time"]

numeric_features = [
    "people_waiting", "active_counters", "avg_service_time", "hour", "day_of_week"
]
categorical_features = ["service_type"]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("category", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=160, random_state=42, max_depth=12
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model trained and saved to", MODEL_PATH)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f} minutes")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")
print("Note: metrics are based on simulated demo data, not real queue observations.")
