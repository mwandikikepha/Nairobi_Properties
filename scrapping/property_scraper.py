"""
Property24 Scraper - Main module
Simple scraper for Nairobi property listings
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from read_scrapped_data import parse_listing_card
import os
from dotenv import load_dotenv

load_dotenv() 


BASE_URL = os.getenv("BASE_URL")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_page(page_num):
#scrape a single page
    url = BASE_URL if page_num == 1 else f"{BASE_URL}?Page={page_num}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"Page {page_num} failed: {e}")
        return []
    
    cards = soup.select(".p24_content")
    
    listings = []
    for card in cards:
        listing = parse_listing_card(card)
        if listing:
            listings.append(listing)
    
    return listings


def scrape_properties(max_pages=200, target=2500):
    #scraping may pages
    all_listings = []
    
    print("Starting scrape...")
    print("=" * 60)
    
    for page in range(1, max_pages + 1):
        print(f"Page {page}...", end=" ")
        
        listings = scrape_page(page)
        
        if listings:
            all_listings.extend(listings)
            print(f"✓ {len(listings)} listings | Total: {len(all_listings)}")
        else:
            print("No listings")
            break
        
        if len(all_listings) >= target:
            break
        
        time.sleep(2)
    
    print("=" * 60)
    print(f"Complete! {len(all_listings)} listings\n")
    
    return all_listings


def save_to_csv(listings, filename='raw_listings.csv'):
    #saving to csv
    if not listings:
        print("No data to save")
        return None
    
    df = pd.DataFrame(listings)
    df.to_csv(filename, index=False)
    
    print(f"Saved: {filename}")
    print(f"Records: {len(df)}\n")
    
    return df


if __name__ == "__main__":
    listings = scrape_properties(max_pages=200, target=2500)
    df = save_to_csv(listings, '../data/raw_listings.csv')
    if df is not None:
        print(df.head())