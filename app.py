import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ⚠️ set_page_config MUST be the very first Streamlit call
st.set_page_config(page_title="AI Life Pattern Analyzer", layout="centered")

# ---------------- LOAD MODEL ----------------
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        st.write("Model not found. Training now...")
        try:
            import model  # runs model.py
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
st.write("Predict your stress level based on your lifestyle habits")

if trained_model is None:
    st.stop()  # Don't render the rest if model failed

# ---------------- INPUT ----------------
# ⚠️ These MUST match the exact columns model.py trained on (in the same order):
# age, sleep_hours, screen_time_hours, stress_level, study_hours,
# physical_activity, caffeine_intake, academic_pressure
# (after dropping student_id, target = last column = academic_pressure)

age = st.slider("Age", 18, 35, 22)
sleep = st.slider("Sleep Hours", 0, 12, 7)
screen = st.slider("Screen Time (hours)", 0, 12, 5)
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
study = st.slider("Study Hours", 0, 12, 4)
physical_activity = st.selectbox("Physical Activity", ["Yes", "No"])
caffeine = st.slider("Caffeine Intake (cups)", 0, 5, 1)

# Encode categoricals the same way model.py does
stress_map = {"Low": 1, "Medium": 2, "High": 3}
activity_map = {"Yes": 1, "No": 0}

stress_encoded = stress_map[stress]
activity_encoded = activity_map[physical_activity]

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    # Shape must match training: 8 features (age, sleep, screen, stress, study, activity, caffeine, academic_pressure)
    # academic_pressure is the TARGET (last col) — so we only pass the first 8 columns
    input_data = np.array([[age, sleep, screen, stress_encoded, study, activity_encoded, caffeine]])
    
    prediction = trained_model.predict(input_data)[0]
    predicted_pressure = round(prediction, 2)

    pressure_label = {1: "Low 🟢", 2: "Medium 🟡", 3: "High 🔴"}
    label = pressure_label.get(round(prediction), f"Score: {predicted_pressure}")

    st.success(f"Predicted Academic Pressure: {label}")

    if prediction >= 2.5:
        st.write("⚠️ High academic pressure detected. Consider reducing screen time and improving sleep.")
    elif prediction >= 1.5:
        st.write("⚖️ Moderate pressure. Maintain balance between study and rest.")
    else:
        st.write("✅ Low academic pressure. Great lifestyle balance!")

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
