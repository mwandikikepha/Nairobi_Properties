"""
Feature Engineering Script
Extracted from day2_cleaning.ipynb
Creates new features from clean data
"""

import pandas as pd


def count_amenities(amenities_str):
    """
    Count number of amenities from string
    
    """
    if pd.isna(amenities_str) or amenities_str == 'None':
        return 0
    return amenities_str.count(',') + 1


def engineer_features(input_path, output_path):
    """
    Create new features from clean data
    
    Features created:
    1. price_per_sqft - Price per square foot
    2. amenity_score - Count of amenities (0-5+)
    3. month - Month from listing date
    
    """

    # Load clean data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} clean listings")
    

    print("FEATURE ENGINEERING started...")

    
    # 1. PRICE PER SQUARE FOOT
    df['price_per_sqft'] = df['price_kes'] / df['size_sqft']
    print("\n Created: price_per_sqft")
    print(f"   Range: KSh {df['price_per_sqft'].min():,.0f} - {df['price_per_sqft'].max():,.0f}")
    print(f"   Median: KSh {df['price_per_sqft'].median():,.0f}")
    

    # 2. AMENITY SCORE

    df['amenity_score'] = df['amenities'].apply(count_amenities)
    print("\n Created: amenity_score")
    print(f"   Distribution:")
    print(df['amenity_score'].value_counts().sort_index())
    

    # 3. MONTH FROM LISTING DATE

    df['listing_date'] = pd.to_datetime(df['listing_date'])
    df['month'] = df['listing_date'].dt.month
    print("\n Created: month")
    print(f"   Months present: {sorted(df['month'].unique())}")
    
    # Save data with features
    df.to_csv(output_path, index=False)
    print(f"\n Saved to: {output_path}")
    print(f" Final: {len(df)} rows, {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    # Run feature engineering
    input_file = "../data/clean_listings.csv"
    output_file = "../data/clean_listings.csv"  # Overwrite with features
    
    df_features = engineer_features(input_file, output_file)
    
