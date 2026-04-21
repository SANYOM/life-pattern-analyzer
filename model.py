import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean column names (IMPORTANT)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Features
X = df[["sleep_hours", "study_hours", "screen_time", "stress_level"]]
y = df["productivity_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
