import streamlit as st
import pandas as pd
import joblib

st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

saved = joblib.load("titanic_survival_model (1).pkl")
model = saved["model"]
encoders = saved["encoders"]
feature_columns = saved["feature_columns"]

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Female", "Male"])
sibling = st.selectbox("Siblings", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox(
    "Embarked",
    ["Southampton", "Cherbourg", "Queenstown"]
)

if st.button("Predict Survival"):
    values = {
        "PClass": pclass,
        "Gender": gender,
        "Sibling": sibling,
        "Embarked": embarked
    }

    try:
        row = {}
        for col in feature_columns:
            row[col] = encoders[col].transform([values[col]])[0]

        prediction = model.predict(pd.DataFrame([row]))[0]
        result = encoders["Survived"].inverse_transform([prediction])[0]

        st.success(f"Predicted survival: {result}")
    except ValueError:
        st.error("One of the selected values was not present in the training data.")
