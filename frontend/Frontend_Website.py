import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import keras

def preprocess_data(precipitation, salinity, evaporation):
    data = pd.DataFrame({
        'Precipitation': [precipitation],
        'Salinity': [salinity],
        'Evaporation': [evaporation]
    })
    
    return scale_data(data)

def scale_data(data):
    min_max_list = [[123.4666667, 58.43333333], [663.114, 189.293], [89.0, 56.0], [17.5, 11.325]]
    for i,col in zip(range(len(min_max_list)),data.columns):
        X=data[col]
        X_scaled = abs((X - min_max_list[i][1]) / (min_max_list[i][0] - min_max_list[i][1]))
        data[col]=X_scaled
    return data

# Function to make prediction using the ML model
def predict(data):
    # Make predictions using the loaded model
    model = keras.models.load_model('model.h5')
    prediction = model.predict(data)
    return prediction

# Streamlit application
def main():
    # Set Streamlit app title
    st.title("ML Prediction with Gauge Meter")

    # Input section
    st.header("Input")
    precipitation = st.number_input("Precipitation", value=0.0)
    salinity = st.number_input("Salinity", value=0.0)
    evaporation = st.number_input("Evaporation", value=0.0)

    # Preprocess input data
    input_data = preprocess_data(precipitation, salinity, evaporation)

    if st.button("Submit"):
        # Preprocess input data
        input_data = preprocess_data(precipitation, salinity, evaporation)

        # Perform prediction
        prediction = predict(input_data)
        print(prediction[0])
        """
        prediction1 = prediction[0]
        prediction1 = prediction1 * 1000
        prediction1 = prediction1 + 35
        prediction[0] = prediction1
        """
        # Output section
        st.subheader("Water Level Prediction")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(prediction[0]),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current Water Level"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': 'darkblue'},
                'steps': [
                    {'range': [0, 0.5], 'color': 'lightgray'},
                    {'range': [0.5, 1], 'color': 'gray'}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig)

# Run the Streamlit application
if __name__ == "__main__":
    main()





