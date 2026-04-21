import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop student_id (not useful for training)
df = df.drop(columns=["student_id"], errors="ignore")

# Map ALL categorical columns properly
stress_map = {"low": 1, "medium": 2, "high": 3}
gender_map = {"male": 0, "female": 1, "other": 2}
yes_no_map = {"yes": 1, "no": 0}
pressure_map = {"low": 1, "medium": 2, "high": 3}

df["gender"] = df["gender"].str.strip().str.lower().map(gender_map)
df["stress_level"] = df["stress_level"].str.strip().str.lower().map(stress_map)
df["physical_activity"] = df["physical_activity"].str.strip().str.lower().map(yes_no_map)
df["academic_pressure"] = df["academic_pressure"].str.strip().str.lower().map(pressure_map)

# Convert remaining columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with any remaining NaN
df = df.dropna()

print(f"✅ Rows after cleaning: {len(df)}")
print("Cleaned Data:\n", df.head())

# Train model
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved successfully!")
