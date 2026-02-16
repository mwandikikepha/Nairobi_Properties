
from convert import (
    parse_price, parse_size, parse_number,
    extract_property_type, extract_amenities, get_current_date
)


def parse_listing_card(card):
# read a single property card and return structured data
    try:
        # Extract elements
        price_tag = card.select_one(".p24_price")
        title_tag = card.select_one(".p24_propertyTitle")
        loc_tag = card.select_one(".p24_location")
        desc_tag = card.select_one(".p24_excerpt")
        
        # Feature details (beds, baths, parking)
        features = card.select(".p24_featureDetails span")
        beds_text = features[0].text.strip() if len(features) > 0 else None
        baths_text = features[1].text.strip() if len(features) > 1 else None
        parking_text = features[2].text.strip() if len(features) > 2 else None
        
        # Size
        size_span = card.select_one(".p24_size span")
        
        # Parse values
        price_kes = parse_price(price_tag.text if price_tag else None)
        title = title_tag.text.strip() if title_tag else None
        location = loc_tag.text.strip() if loc_tag else "Nairobi"
        description = desc_tag.text.strip() if desc_tag else ""
        
        bedrooms = parse_number(beds_text)
        bathrooms = parse_number(baths_text)
        size_sqft = parse_size(size_span.text if size_span else None)
        
        property_type = extract_property_type(title)
        amenities = extract_amenities(description, parking_text)
        
        # Only return if has minimum required data
        if price_kes and bedrooms is not None:
            return {
                "location": location,
                "property_type": property_type,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms if bathrooms else 1,
                "size_sqft": size_sqft,
                "amenities": amenities,
                "price_kes": price_kes,
                "listing_date": get_current_date()
            }
        
        return None
    
    except Exception as e:
        print(f"Error parsing card: {e}")
        return None