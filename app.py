import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model and encoders
# -----------------------------
model = joblib.load("satisfaction_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
customer_type_encoder = joblib.load("customer_type_encoder.pkl")
satisfaction_encoder = joblib.load("satisfaction_encoder.pkl")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Satisfaction Prediction",
    page_icon="😊",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("😊 Customer Satisfaction Prediction")
st.write("Enter the customer details below to predict satisfaction.")

st.divider()

# -----------------------------
# Input Fields
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    gender_encoder.classes_
)

customer_type = st.selectbox(
    "Customer Type",
    customer_type_encoder.classes_
)

customer_service_rating = st.slider(
    "Customer Service Rating",
    min_value=1,
    max_value=5,
    value=3
)

complaint_count = st.number_input(
    "Complaint Count",
    min_value=0,
    max_value=20,
    value=0
)

previous_purchases = st.number_input(
    "Previous Purchases",
    min_value=0,
    max_value=100,
    value=5
)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("🔮 Predict Satisfaction"):

    # Encode categorical inputs
    gender_encoded = gender_encoder.transform([gender])[0]

    customer_type_encoded = customer_type_encoder.transform(
        [customer_type]
    )[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Customer_Type": [customer_type_encoded],
        "Customer_Service_Rating": [customer_service_rating],
        "Complaint_Count": [complaint_count],
        "Previous_Purchases": [previous_purchases]
    })

    # Prediction
    prediction = model.predict(input_data)

    # Convert prediction back to original label
    result = satisfaction_encoder.inverse_transform(prediction)[0]

    # -----------------------------
    # Display Result
    # -----------------------------

    st.subheader("Prediction Result")

    if str(result).lower() in ["satisfied", "yes", "positive", "1"]:
        st.success(f"😊 Customer is **{result}**")
    else:
        st.error(f"😞 Customer is **{result}**")

    # Prediction probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        confidence = max(probability) * 100

        st.info(f"Prediction Confidence: **{confidence:.2f}%**")