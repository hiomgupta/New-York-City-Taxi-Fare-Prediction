import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from geopy.distance import geodesic
from xgboost import XGBRegressor

# Load the model
model = joblib.load('nyc_taxi_model.pkl')  # Ensure the model file is saved in the repository

# Define popular landmarks and their coordinates
landmarks = {
    'jfk': (-73.7781, 40.6413),
    'lga': (-73.8740, 40.7769),
    'ewr': (-74.1745, 40.6895),
    'met': (-73.9632, 40.7794),
    'wtc': (-74.0134, 40.7128)
}

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])

def preprocess_input(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    pickup_datetime = pd.to_datetime(pickup_datetime)
    year = pickup_datetime.year
    month = pickup_datetime.month
    day = pickup_datetime.day
    weekday = pickup_datetime.weekday()
    hour = pickup_datetime.hour
    trip_distance = haversine_np(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

    data = {
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'passenger_count': [passenger_count],
        'pickup_datetime_year': [year],
        'pickup_datetime_month': [month],
        'pickup_datetime_day': [day],
        'pickup_datetime_weekday': [weekday],
        'pickup_datetime_hour': [hour],
        'trip_distance': [trip_distance]
    }
    input_df = pd.DataFrame(data)
    for landmark, coords in landmarks.items():
        add_landmark_dropoff_distance(input_df, landmark, coords)
    return input_df

def main():
    st.title("NYC Taxi Fare Prediction")
    st.markdown("This app predicts the taxi fare in NYC based on input features.")
    
    pickup_datetime = st.text_input("Pickup DateTime", "2013-07-06 17:18:00")
    pickup_longitude = st.number_input("Pickup Longitude", -73.95)
    pickup_latitude = st.number_input("Pickup Latitude", 40.75)
    dropoff_longitude = st.number_input("Dropoff Longitude", -73.99)
    dropoff_latitude = st.number_input("Dropoff Latitude", 40.75)
    passenger_count = st.number_input("Passenger Count", 1, 10, 1)

    if st.button("Predict Fare"):
        input_df = preprocess_input(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count)
        prediction = model.predict(input_df)
        st.success(f"Estimated Fare: ${prediction[0]:.2f}")

if __name__ == '__main__':
    main()
