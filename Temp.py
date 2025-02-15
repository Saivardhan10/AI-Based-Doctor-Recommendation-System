import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset from CSV
df = pd.read_csv("Test.csv")

# Ensure necessary columns exist
if "Symptoms" not in df.columns or "Specialist" not in df.columns:
    raise ValueError("Dataset must contain 'Symptoms' and 'Specialist' columns")

# Convert symptoms column to list format and normalize text to lowercase
df["Symptoms"] = df["Symptoms"].apply(lambda x: x.lower().split(", "))

# One-hot encoding for symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])

# Encoding specialist labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Specialist"])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training using RandomForest for efficiency
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict specialist
def recommend_doctor(symptoms):
    symptoms_list = symptoms.lower().split(", ")  # Convert input to lowercase
    input_vector = mlb.transform([symptoms_list])
    prediction = rf_model.predict(input_vector)
    recommended_specialist = label_encoder.inverse_transform(prediction)[0]
    return recommended_specialist

# Streamlit interface
st.title("AI-Based Doctor Recommendation System")
st.write(f"Model Accuracy: {accuracy:.2f}")

user_symptoms = st.text_input("Enter your symptoms (comma separated):").lower()
if st.button("Get Recommendation"):
    if user_symptoms:
        specialist = recommend_doctor(user_symptoms)
        st.success(f"Recommended Specialist: {specialist}")
    else:
        st.warning("Please enter symptoms to get a recommendation.")
