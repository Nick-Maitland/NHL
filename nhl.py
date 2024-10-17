import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('my_trained_model.h5')

# Title of the app
st.title("NHL Game Outcome Predictor")

# User input section
st.header("Input the game statistics")

# Create input fields for the features used in training the model
goals_x = st.number_input("Goals for Team X", min_value=0, max_value=10, value=1)
assists = st.number_input("Assists", min_value=0, max_value=10, value=1)
shots_x = st.number_input("Shots for Team X", min_value=0, max_value=50, value=10)
hits_x = st.number_input("Hits for Team X", min_value=0, max_value=50, value=5)
goals_y = st.number_input("Goals for Team Y", min_value=0, max_value=10, value=1)
shots_y = st.number_input("Shots for Team Y", min_value=0, max_value=50, value=10)
powerPlayGoals_x = st.number_input("Power Play Goals for Team X", min_value=0, max_value=10, value=1)

# Prepare the input data for the model
input_data = np.array([[goals_x, assists, shots_x, hits_x, goals_y, shots_y, powerPlayGoals_x]])

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    prediction = model.predict(input_data)

    # Convert the prediction to a binary result (won or lost)
    prediction_result = "Won" if prediction[0] > 0.5 else "Lost"
    st.write(f"Predicted Outcome: {prediction_result}")
