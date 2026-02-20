"""
Data Cleaning Script
Extracted from day2_cleaning.ipynb
"""

import pandas as pd


def clean_raw_data(input_path, output_path):
    """
    Clean raw property listings
    
    Steps:
    1. Remove duplicates
    2. Filter price outliers
    3. Fix and filter size outliers
    4. Remove extreme properties
    5. Handle missing values
    6. Standardize text
    
    Args:
        input_path: Path to raw_listings.csv
        output_path: Path to save clean_listings.csv
    
    Returns:
        DataFrame: Cleaned data
    """
    
    # Load data
    df = pd.read_csv(input_path)
    df_clean = df.copy()
    print(f"Starting rows: {len(df_clean)}")
    

    # REMOVING DUPLICATES
    df_clean = df_clean.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df_clean)}")
    print(f"Duplicates after: {df_clean.duplicated().sum()}")
    
    # REMOVING PRICE OUTLIERS
    df_clean = df_clean[(df_clean['price_kes'] >= 1_000_000) & 
                        (df_clean['price_kes'] <= 500_000_000)]
    print(f"\nAfter price filtering (1M-500M): {len(df_clean)} rows remain")
    

    # SIZE CLEANING

    # Count outliers before removing
    small_count = (df_clean['size_sqft'] < 200).sum()
    large_count = (df_clean['size_sqft'] > 20000).sum()

    # Remove small and large outliers directly
    df_clean = df_clean[(df_clean['size_sqft'] >= 200) & (df_clean['size_sqft'] <= 20000)]

    print(f"Removed {small_count} very small properties (<200 sqft)")
    print(f"Removed {large_count} very large properties (>20,000 sqft)")
    print(f"After size filtering: {len(df_clean)} rows")
    
    df_clean = df_clean[df_clean['bedrooms'] <= 8]
    print(f"After bedroom validation: {len(df_clean)} rows")

    # Handle missing sizes
    missing_sizes = df_clean['size_sqft'].isna().sum()
    print(f"\nRows with missing sizes: {missing_sizes}")
    df_clean = df_clean.dropna(subset=['size_sqft'])
    print(f"After dropping missing sizes: {len(df_clean)} rows")
    
    # ADDITIONAL PRICE CLEANING
    
    print(f"\nBefore additional cleaning: {len(df_clean)} rows")
    
    # Remove extreme luxury properties
    df_clean = df_clean[df_clean['price_kes'] <= 100_000_000]
    print(f"After removing >100M: {len(df_clean)} rows")
    
    # Remove very cheap properties
    df_clean = df_clean[df_clean['price_kes'] >= 5_000_000]
    print(f"After removing <5M: {len(df_clean)} rows")
    
    
    # SORTING AMENITIES
    
    # Drop properties with no amenities listed
    df_clean = df_clean.dropna(subset=['amenities'])
    print(f"After dropping missing amenities: {len(df_clean)} rows")
    

    # STANDARDIZING TEXT

    df_clean['location'] = df_clean['location'].str.title().str.strip()
    df_clean['property_type'] = df_clean['property_type'].str.title().str.strip()
    
    
    print(f"\nCleaning complete Kept {len(df_clean)} listings")
    
    # Save clean data
    df_clean.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return df_clean


if __name__ == "__main__":
    # Run cleaning
    input_file = "../data/raw_listings.csv"
    output_file = "../data/clean_listings.csv"
    
    df_clean = clean_raw_data(input_file, output_file)
    
    print(f"\n Final: {len(df_clean)} rows, {df_clean.shape[1]} columns")