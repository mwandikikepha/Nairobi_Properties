# Nairobi House Price Data Dictionary

## Dataset Overview

- **Total Listings**: 500 properties
- **Collection Date**: February 16, 2026
- **Source**: Property24
- **Locations Covered**: 15+ Nairobi neighborhoods
- **Property Types**: Apartments (452), Houses (29), Townhouses (19)

## Column Descriptions

| Column | Type | Description | Examples | Stats/Notes |
|--------|------|-------------|----------|--------------|
| **location** | string | Nairobi neighborhood | Westlands, Kilimani, Kileleshwa | 15 unique areas, Westlands dominates (52%) |
| **property_type** | string | Type of property | Apartment, House, Townhouse | 90% apartments, 10% houses/townhouses |
| **bedrooms** | integer | Number of bedrooms | 1-9 | Range: 1-9, Most common: 1 (161), 2 (155) |
| **bathrooms** | integer | Number of bathrooms | 1-9 | Often matches bedrooms, sometimes more |
| **size_sqft** | float | Size in square feet | 32.29 - 43,099 | 18 missing values, has extreme outliers |
| **amenities** | string | Features included | "Parking (1)", NaN, "Parking (2), Garden" | 6 missing, mostly parking info only |
| **price_kes** | integer | Price in Kenyan Shillings | 360,000 - 350,000,000 | Mean: 23.6M, Median: 12.8M |
| **listing_date** | date | When listed | 2026-02-16 | All same date (scraped today) |

## Data Quality Issues Identified

### Critical Issues
1. **Duplicates**: There are duplicate rows 
2. **Price outliers**: 360,000 KES property in Runda - impossibly low
3. **Size outliers**: 32 sqft properties with 4 bedrooms - impossible
4. **Missing values**: 18 missing sizes, 6 missing amenities


## Location Summary

| Location | Count | Price Range (KES) | Notes |
|----------|-------|-------------------|-------|
| **Westlands** | 262 (52%) | 5.3M - 35M | Most data, good sample |
| **Kilimani** | 87 (17%) | 5.5M - 54M | Premium area |
| **Kileleshwa** | 47 (9%) | 5.1M - 48M | Middle-high |
| **Syokimau** | 22 (4%) | 4.8M - 12.5M | More affordable |
| **Lavington** | 13 (3%) | 5.2M - 220M | Luxury mixed with regular |
| **Runda** | 10 (2%) | 360K - 260M | Has price error (360K) |
| **Riverside** | 9 (2%) | 7.2M - 28M | Premium |
| **Kitisuru** | 8 (2%) | 60M - 350M | Luxury, has size errors |
| **Karen** | 5 (1%) | 105M - 250M | Luxury |
| **Others** | 37 (7%) | Various | Parklands, Spring Valley, Kiambu Road, Lower Kabete, Muthangari, Ridgeways, Nyari, Kyuna, Peponi, Loresho, South C |

## Property Type Breakdown

| Type | Count | Avg Price | Typical Locations |
|------|-------|-----------|-------------------|
| **Apartment** | 452 | ~18M | Westlands, Kilimani, Kileleshwa |
| **House** | 29 | ~85M | Runda, Karen, Kitisuru |
| **Townhouse** | 19 | ~70M | Lavington, Runda, Spring Valley |

## Day 2 Cleaning Plan

1. Remove duplicates
2. Fix or remove 360,000 KES price error
3. Remove sizes < 300 sqft 
4. Remove sizes > 10,000 sqft 
5. Drop rows with missing size_sqft
6. Fill missing amenities with "None"
7. Create price_per_sqft feature
8. Extract parking count from amenities
9. Create has_garden flag

## Expected Data After Cleaning

| Metric | Raw Data | After Cleaning |
|--------|----------|----------------|
| **Rows** | 500 | 400 |
| **Duplicates** | 100 | 0 |
| **Missing sizes** | 18 | 0 |
| **Price outliers** | Several | Removed/Fixed |
| **Size outliers** | Many | Removed |


# Data Dictionary - Engineered Features 

## New Features Created (Day 2)

The following features were created during data cleaning and feature engineering:

| Column Name | Description | Data Type | Example Values | Calculation |
|------------|-------------|-----------|----------------|-------------|
| **price_per_sqft** | Price per square foot | Float (decimal) | 12000.50, 8500.00 | `price_kes / size_sqft` |
| **amenity_score** | Count of amenities | Integer | 0, 1, 2, 3, 4, 5 | Count commas in amenities + 1 |
| **month** | Month from listing date | Integer | 1, 2, 3...12 | Extracted from `listing_date` |

---

## Feature Details

### price_per_sqft
- **Purpose:** Normalize prices by property size
- **Use Case:** Compare value across different sized properties
- **Typical Range:** 5,000 - 50,000 KES per sqft
- **Missing Values:** Same as size_sqft (property has no size data)

### amenity_score
- **Purpose:** Quantify amenity richness
- **Calculation:** "Parking, Pool, Gym" → count = 3
- **Range:** 0 (no amenities) to 5+ (luxury properties)
- **Use Case:** Feature for price prediction model

### month
- **Purpose:** Capture seasonal trends in pricing
- **Range:** 1-12 (January to December)
- **Use Case:** Identify if certain months have higher/lower prices

---

**Created:** February 17, 2026  
**Used in:** Day 3+ modeling