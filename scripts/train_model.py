"""
Train Random Forest model for house price prediction
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(input_file, output_model):
    
    print("TRAINING RANDOM FOREST MODEL.............")

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    le_location = LabelEncoder()
    df['location_encoded'] = le_location.fit_transform(df['location'])
    
    le_property = LabelEncoder()
    df['property_type_encoded'] = le_property.fit_transform(df['property_type'])
    
    # Features and target
    feature_columns = ['bedrooms', 'bathrooms', 'size_sqft', 'location_encoded', 
                       'property_type_encoded', 'amenity_score']
    
    X = df[feature_columns]
    y = df['price_kes']
    print(f"Features: {feature_columns}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")
    
    # Train Random Forest
    print("\nTraining Random Forest.......")
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    print("Training complete")
    
    # Predict and evaluate
    print("\n Evaluating model...")
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRandom Forest Performance:")
    print(f"  MAE: {mae/1_000_000:.2f}M KES")
    print(f"  R²:  {r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop Features:")
    for _, row in importance.iterrows():
        print(f"  {row['Feature']:20s}: {row['Importance']*100:.1f}%")
    
    # Save model
    print(f"\nSaving model to: {output_model}")
    with open(output_model, 'wb') as f:
        pickle.dump(rf_model, f)
    print("Model saved successfully")
    

    print("TRAINING COMPLETED...............")
    
    return rf_model, {'mae': mae, 'r2': r2, 'importance': importance}

if __name__ == "__main__":
    # Run training
    input_file = "../data/clean_listings.csv"
    output_model = "../models/model.pkl"
    
    model, metrics = train_model(input_file, output_model)
    
    print(f"\nFinal model: MAE = {metrics['mae']/1_000_000:.2f}M KES, R² = {metrics['r2']:.3f}")