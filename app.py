try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.subheader("Patient attributes")

with st.form("heart_form"):
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex_label = st.selectbox("Sex (UCI code)", ["Female (0)", "Male (1)"])
    cp_label = st.selectbox(
        "Chest pain type (cp)",
        [
            "1 — Typical angina",
            "2 — Atypical angina",
            "3 — Non-anginal pain",
            "4 — Asymptomatic",
        ],
    )
    thalach = st.number_input(
        "Maximum heart rate achieved (thalach)",
        min_value=50,
        max_value=250,
        value=150,
    )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    sex = 0 if sex_label.startswith("Female") else 1
    cp = int(cp_label.split(" ")[0])

    # These column names and order must match the model training notebook.
    input_df = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"],
    )

    try:
        prediction = int(model.predict(input_df)[0])
        if prediction == 1:
            st.warning(
                "The model predicts the disease-present class for these inputs."
            )
        else:
            st.success(
                "The model predicts the no-disease class for these inputs."
            )

        st.caption(
            "This output is for coursework demonstration only. It cannot confirm "
            "or rule out heart disease; seek a qualified healthcare professional "
            "for medical assessment."
        )
    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the saved model was trained with "
            "the four features age, sex, cp, thalach in that order. "
            f"Technical details: {exc}"
        )
