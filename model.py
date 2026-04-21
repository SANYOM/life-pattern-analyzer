import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop student_id only
df = df.drop(columns=["student_id"], errors="ignore")

# Map ALL categorical columns
df["gender"] = df["gender"].str.strip().str.lower().map({"male": 0, "female": 1, "other": 2})
df["stress_level"] = df["stress_level"].str.strip().str.lower().map({"low": 1, "medium": 2, "high": 3})
df["physical_activity"] = df["physical_activity"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["academic_pressure"] = df["academic_pressure"].str.strip().str.lower().map({"low": 1, "medium": 2, "high": 3})

# Convert all to numeric, drop bad rows
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

print(f"✅ Rows after cleaning: {len(df)}")
print("Columns:", list(df.columns))

# Train
X = df.iloc[:, :-1]  # 8 features
y = df.iloc[:, -1]   # academic_pressure

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved!")
