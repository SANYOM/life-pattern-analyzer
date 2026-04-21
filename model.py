import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 🔥 Convert categorical values to numbers
mapping = {
    "low": 1,
    "medium": 2,
    "high": 3
}

for col in df.columns:
    df[col] = df[col].replace(mapping)

# Convert everything to numeric (important)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna()

print("Cleaned Data:\n", df.head())

# Use generic approach
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully!")
