import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Life Pattern Analyzer", layout="centered")

def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        st.write("Model not found. Training now...")
        try:
            import model
        except Exception as e:
            st.error(f"Model training failed: {e}")
            return None
        if os.path.exists("model.pkl"):
            return joblib.load("model.pkl")
        else:
            st.error("model.pkl was not created!")
            return None

trained_model = load_model()

st.title("🧠 AI Life Pattern Analyzer")
st.write("Predict your academic pressure based on your lifestyle habits")

if trained_model is None:
    st.stop()

# ---------------- INPUTS (must match EXACT column order from model.py) ----------------
# Order: age, gender, sleep_hours, screen_time_hours, stress_level, study_hours, physical_activity, caffeine_intake

age = st.slider("Age", 18, 35, 22)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender_map = {"male": 0, "female": 1, "other": 2}
gender_encoded = gender_map[gender.lower()]

sleep = st.slider("Sleep Hours", 0, 12, 7)
screen = st.slider("Screen Time (hours)", 0, 12, 5)

stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
stress_map = {"low": 1, "medium": 2, "high": 3}
stress_encoded = stress_map[stress.lower()]

study = st.slider("Study Hours", 0, 12, 4)

physical_activity = st.selectbox("Physical Activity", ["Yes", "No"])
activity_encoded = 1 if physical_activity == "Yes" else 0

caffeine = st.slider("Caffeine Intake (cups)", 0, 5, 1)

# ---------------- PREDICTION ----------------
if st.button("Predict Academic Pressure"):
    # EXACT same order as training columns (after dropping student_id, before academic_pressure)
    input_data = np.array([[age, gender_encoded, sleep, screen, stress_encoded, study, activity_encoded, caffeine]])

    prediction = trained_model.predict(input_data)[0]

    pressure_label = {1: "Low 🟢", 2: "Medium 🟡", 3: "High 🔴"}
    label = pressure_label.get(round(prediction), f"Score: {round(prediction, 2)}")

    st.success(f"Predicted Academic Pressure: {label}")

    if prediction >= 2.5:
        st.write("⚠️ High pressure detected. Improve sleep and reduce screen time.")
    elif prediction >= 1.5:
        st.write("⚖️ Moderate pressure. Stay consistent with your habits.")
    else:
        st.write("✅ Low pressure. Great lifestyle balance!")

# ---------------- VISUALIZATION ----------------
st.subheader("📊 Sample Weekly Productivity Trend")
sample_data = pd.DataFrame({
    "Days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "Productivity": np.random.randint(3, 10, 7)
})
fig, ax = plt.subplots()
ax.plot(sample_data["Days"], sample_data["Productivity"], marker="o", color="royalblue")
ax.set_xlabel("Day")
ax.set_ylabel("Productivity Score")
ax.set_title("Weekly Trend")
st.pyplot(fig)
