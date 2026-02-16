
#Usage: python run_scraper.py

from property_scraper import scrape_properties, save_to_csv

# Scrape
listings = scrape_properties(max_pages=25, target=500)

# Save
df = save_to_csv(listings, '../data/raw_listings.csv')

# Show sample
if df is not None:
    print("Sample data:")
    print(df.head(3))
    
    print(f"\nLocations: {df['location'].unique()}")
    print(f"Price range: KSh {df['price_kes'].min():,.0f} - {df['price_kes'].max():,.0f}")