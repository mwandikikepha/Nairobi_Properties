"""
Data parsing utilities for Property24 scraper
"""

import re
from datetime import datetime


def parse_price(price_text):
    """Extract numeric price from 'KSh 14 000 000'"""
    if not price_text:
        return None
    numbers = re.sub(r'[^\d]', '', price_text)
    return int(numbers) if numbers else None


def parse_size(size_text):
    """Convert size from m² to sqft"""
    if not size_text:
        return None
    match = re.search(r'(\d+)\s*m²', size_text)
    if match:
        sqm = int(match.group(1))
        return round(sqm * 10.764, 2)
    return None


def parse_number(text):
    """Extract first number from text like '3 Bedrooms' -> 3"""
    if not text:
        return None
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else None


def extract_property_type(title):
    """Determine property type from title"""
    if not title:
        return "Apartment"
    
    title_lower = title.lower()
    if 'house' in title_lower and 'townhouse' not in title_lower:
        return "House"
    elif 'townhouse' in title_lower:
        return "Townhouse"
    elif 'land' in title_lower or 'plot' in title_lower:
        return "Land"
    elif 'villa' in title_lower:
        return "Villa"
    else:
        return "Apartment"


def extract_amenities(description, parking):
    """Extract amenities from description and parking info"""
    amenities = []
    
    desc_lower = (description or "").lower()
    
    # Parking
    if parking:
        park_num = parse_number(parking)
        amenities.append(f"Parking ({park_num})" if park_num else "Parking")
    
    # Other amenities
    if 'pool' in desc_lower or 'swimming' in desc_lower:
        amenities.append("Pool")
    if 'gym' in desc_lower or 'fitness' in desc_lower:
        amenities.append("Gym")
    if 'security' in desc_lower or 'gated' in desc_lower:
        amenities.append("Security")
    if 'garden' in desc_lower:
        amenities.append("Garden")
    
    return ", ".join(amenities) if amenities else "None"


def get_current_date():
    """Get current date in YYYY-MM-DD format"""
    return datetime.now().strftime('%Y-%m-%d')