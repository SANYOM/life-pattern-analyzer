import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("dataset.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(df.columns)

# Adjust these after checking print
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column as target

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
