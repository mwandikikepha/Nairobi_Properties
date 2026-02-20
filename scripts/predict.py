"""
Make price predictions using trained model
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model(model_path="../models/model.pkl"):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def prepare_input(bedrooms, bathrooms, size_sqft, location, property_type, amenity_score=0):
    
    # Load original data to get encodings (we need the fitted encoders)
    df = pd.read_csv("../data/clean_listings.csv")
    
    # Fit encoders on original data (same as training)
    le_location = LabelEncoder()
    le_location.fit(df['location'])
    
    le_property = LabelEncoder()
    le_property.fit(df['property_type'])
    
    # Encode inputs
    try:
        loc_encoded = le_location.transform([location])[0]
    except:
        # If location not seen before, use most common
        loc_encoded = le_location.transform([df['location'].mode()[0]])[0]
        print(f" Location '{location}' not in training data. Using '{df['location'].mode()[0]}' instead.")
    
    try:
        prop_encoded = le_property.transform([property_type])[0]
    except:
        prop_encoded = le_property.transform(['Apartment'])[0]
        print(f" Property type '{property_type}' not recognized. Using 'Apartment'.")
    
    # Create feature array
    features = pd.DataFrame([[
        bedrooms, bathrooms, size_sqft, loc_encoded, prop_encoded, amenity_score
    ]], columns=['bedrooms', 'bathrooms', 'size_sqft', 'location_encoded', 
                 'property_type_encoded', 'amenity_score'])
    
    return features

def predict_price(model, features):
    """Make prediction"""
    prediction = model.predict(features)[0]
    return prediction

def get_price_range(prediction, mae=4_930_000):
    
    lower = prediction - mae
    upper = prediction + mae
    return lower, upper

def interpret_prediction(prediction, bedrooms, location):
    
    print("PRICE PREDICTION..................")
    print(f"\n {bedrooms}-bedroom property in {location}")
    print(f"\n Predicted Price: KSh {prediction:,.0f}")
    print(f"   ({prediction/1_000_000:.2f} Million KES)")
    
    lower, upper = get_price_range(prediction)
    print(f"\n Estimated Range: KSh {lower:,.0f} - {upper:,.0f}")
    print(f"   (Based on model MAE of ±4.93M)")
    

if __name__ == "__main__":

    model = load_model()
    
    # Test a single property
    print("\n Testing single prediction:")
    features = prepare_input(
        bedrooms=3,
        bathrooms=4,
        size_sqft=1244,
        location='Westlands',
        property_type='Apartment',
        amenity_score=1
    )
    
    prediction = predict_price(model, features)
    interpret_prediction(prediction, 3, 'Westlands')
    
   