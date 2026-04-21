import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Features (edit based on your dataset columns)
X = df[["sleep_hours", "study_hours", "screen_time", "stress_level"]]
y = df["productivity_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")
