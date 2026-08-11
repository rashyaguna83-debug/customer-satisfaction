import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("satisfaction.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

# Remove duplicate records
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mode().iloc[0])

# Display column names
print("\nColumns in Dataset:")
print(df.columns.tolist())

# Encode Gender
gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"])

# Encode Customer Type
customer_type_encoder = LabelEncoder()
df["Customer_Type"] = customer_type_encoder.fit_transform(
    df["Customer_Type"]
)

# Encode Satisfaction (Target)
satisfaction_encoder = LabelEncoder()
df["Satisfaction"] = satisfaction_encoder.fit_transform(
    df["Satisfaction"]
)

# Features
X = df[
    [
        "Age",
        "Gender",
        "Customer_Type",
        "Customer_Service_Rating",
        "Complaint_Count",
        "Previous_Purchases"
    ]
]

# Target
y = df["Satisfaction"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=satisfaction_encoder.classes_
    )
)

# Save model
joblib.dump(model, "satisfaction_model.pkl")

# Save encoders
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(customer_type_encoder, "customer_type_encoder.pkl")
joblib.dump(satisfaction_encoder, "satisfaction_encoder.pkl")

print("\nModel saved successfully!")import streamlit as st
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