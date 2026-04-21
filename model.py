import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns:", df.columns)

# Adjust these if needed
X = df[["sleep_hours", "study_hours", "screen_time", "stress_level"]]
y = df["productivity_score"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model saved!")
