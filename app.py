import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Life Pattern Analyzer", layout="centered")

st.title("🧠 AI Life Pattern Analyzer")
st.write("Predict your productivity based on lifestyle habits")

# ---------------- INPUT ----------------
sleep = st.slider("Sleep Hours", 0, 12, 7)
study = st.slider("Study Hours", 0, 12, 4)
screen = st.slider("Screen Time", 0, 12, 5)
stress = st.slider("Stress Level", 1, 10, 5)

# ---------------- PREDICTION ----------------
if st.button("Predict Productivity"):
    input_data = np.array([[sleep, study, screen, stress]])
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Productivity Score: {round(prediction,2)}")

    # Simple insight logic
    if prediction > 7:
        st.write("🔥 High productivity! Keep it up.")
    elif prediction > 4:
        st.write("⚖️ Moderate productivity. Improve consistency.")
    else:
        st.write("⚠️ Low productivity. Reduce screen time & stress.")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Sample Trend Visualization")

sample_data = pd.DataFrame({
    "Days": range(1, 8),
    "Productivity": np.random.randint(3, 10, 7)
})

fig, ax = plt.subplots()
ax.plot(sample_data["Days"], sample_data["Productivity"])
ax.set_xlabel("Days")
ax.set_ylabel("Productivity")

st.pyplot(fig)
