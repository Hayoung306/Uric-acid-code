import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# ======================
# 📌 Load Model
# ======================
@st.cache_resource
def load_model():
    return joblib.load("dua_xgb_model_subject_5.pkl")

# ======================
# 🎯 Streamlit UI
# ======================
st.title("Tear Uric Acid Predictor (Event Chaining with Saturation Decay)")
st.write("Each event starts from the last event's final UA. Inside each, UA = start_UA + ΔUA(t), with saturation control.")

st.subheader("Daily Event Schedule")


num_events = st.number_input("Number of Events", min_value=1, max_value=8, value=8)
initial_ua = st.number_input("Initial Uric Acid Level (μM)", min_value=0.0, value=75.0)
lag_time = st.number_input("Diffusion Lag Time (min)", min_value=0, max_value=30, value=0)


available_times = [datetime.strptime(f"{hour}:00", "%H:%M").time() for hour in range(8, 24, 2)]

plan = []
for i in range(num_events):
    st.markdown(f"**Event {i+1}**")
    default_time_index = i if i < len(available_times) else 0
    time = st.selectbox(f"Time", available_times, index=default_time_index, key=f"time_{i}")
    situation = st.selectbox(f"Situation", ["Diet", "Exercise", "Rest"], key=f"situation_{i}")
    if situation == "Diet":
        detail = st.selectbox(f"Detail", ["Low Purine Diet", "High Purine Diet"], key=f"detail_{i}")
    elif situation == "Exercise":
        detail = st.selectbox(f"Detail", ["Jogging", "Walking"], key=f"detail_{i}")
    else:
        detail = st.selectbox(f"Detail", ["Sitting", "Lying"], key=f"detail_{i}")
    plan.append({
        "time": time.strftime("%H:%M"),
        "situation": situation,
        "detail": detail
    })

# ======================
# 🚀 Prediction
# ======================
if st.button("Predict"):
    model = load_model()
    interval = 5
    duration = 120
    current_ua = initial_ua
    timeline = []

    max_ua = 150
    tau = 15

    for i, event in enumerate(plan):
        event_start = datetime.strptime(event["time"], "%H:%M")

        for step in range(0, duration + 1, interval):
            now = event_start + timedelta(minutes=step)
            adjusted_now = now - timedelta(minutes=lag_time)

            input_row = pd.DataFrame([{
                "Situation": event["situation"],
                "Detail Situation": event["detail"],
                "Minute Elapsed": max(0, int((adjusted_now - event_start).total_seconds() // 60)),
                "Set": 1
            }])

            delta = model.predict(input_row)[0]

            if current_ua > max_ua:
                decay_factor = np.exp(-(current_ua - max_ua) / tau)
                delta *= decay_factor

            ua = current_ua + delta

            timeline.append({
                "Clock": now,
                "ClockStr": now.strftime("%H:%M"),
                "Event": f"{i+1}",
                "ΔUA": round(delta, 3),
                "Predicted UA (μM)": round(ua, 2)
            })

        current_ua = ua

    df = pd.DataFrame(timeline).drop_duplicates(subset="Clock").sort_values("Clock")

    # ======================
    # 📈 Visualization
    # ======================
    st.subheader("📈 Tear Uric Acid Over Time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Clock"], df["Predicted UA (μM)"], marker='o')
    ax.set_xticks(df["Clock"][::6])
    ax.set_xticklabels([d.strftime("%H:%M") for d in df["Clock"][::6]], rotation=45)
    ax.set_ylabel("Tear UA (μM)")
    ax.set_ylim(70, 130)
    ax.set_title("Predicted Tear UA (Event-wise ΔUA, with Saturation Decay)")
    ax.grid(True)
    st.pyplot(fig)

    # ======================
    # 💾 CSV Export
    # ======================
    st.subheader("Download Prediction Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", csv, "predicted_tear_ua.csv", "text/csv")
