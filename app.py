
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"


@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


st.title("Heart Disease Prediction System")
st.write(
    "Enter the four selected attributes to get a model prediction. "
    "This application is for educational purposes only."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Upload heart_disease_model.pkl "
        "to the same repository folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("Patient Attributes")

with st.form("heart_disease_form"):
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=50
    )

    sex_label = st.selectbox(
        "Sex", ["Female (0)", "Male (1)"]
    )

    cp = st.selectbox(
        "Chest pain type (cp)", [1, 2, 3, 4]
    )

    thalach = st.number_input(
        "Maximum heart rate achieved (thalach)",
        min_value=50,
        max_value=250,
        value=150,
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    sex = 0 if sex_label.startswith("Female") else 1

    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"],
    )

    try:
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            st.warning(
                "Prediction: Disease-present class"
            )
        else:
            st.success(
                "Prediction: No-disease class"
            )

    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the model was trained "
            "using these four features in this order: "
            "age, sex, cp, thalach. "
            f"Details: {exc}"
        )

st.caption(
    "This is a student ML project, not a medical diagnostic tool."
)
