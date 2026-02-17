# Nairobi House Price Prediction

**6-Day Intensive Sprint - Prop-Tech MVP**

A machine learning project to predict house prices in Nairobi, Kenya using data from Nairobi Price Listings.

---

##  Project Status

**Current Phase:** Day 2 Complete 

-  Day 1: Data Collection & Structuring
-  Day 2: Data Cleaning & Feature Engineering
-  Day 3: Exploratory Analysis & Baseline Model
-  Day 4: Model Improvement & Explainability
-  Day 5: Build Pricing App (Deployment)
-  Day 6: Dashboard & Final Presentation

---

##  Project Structure

```
nairobi_property/
├── data/
│   ├── raw_listings.csv           # Original scraped data 
│   ├── clean_listings.csv         # Cleaned data (ready for modeling)
│   └── data_dictionary.md         # Data documentation
├── notebooks/
│   ├── day1.ipynb                 # Data collection notebook
│   ├── day2_cleaning.ipynb        # Data cleaning & feature engineering
│   └── day2_eda.ipynb            # Exploratory data analysis
├── scrapping/
│   ├── convert.py                 # Data conversion
│   ├── read_scrapped_data.py     # HTML parser
│   ├── property_scraper.py        # Main scraper
│   └── run_scrapper.py           # Scraper runner
└── README.md
```

---

##  Dataset Overview

**Source:** Scrapping websites with property data 
**Collection Date:** February 16, 2026  
**Raw Data:** 801 listings  
**Clean Data:** >700 listings (after removing outliers)

### Key Features:
- **Location** - Nairobi neighborhoods 
- **Property Type** - Apartment, House, Townhouse, Villa
- **Bedrooms** - 1-6 bedrooms
- **Bathrooms** - 1-5 bathrooms
- **Size** - Square footage (converted from m²)
- **Amenities** - Parking, Pool, Gym, Security, Garden
- **Price** - Kenyan Shillings (KES)

### Engineered Features:
- `price_per_sqft` - Price per square foot
- `amenity_score` - Count of amenities (0-5)
- `month` - Listing month for temporal analysis

---

##  Key Insights (Day 2)

### 1. **Most Expensive Locations**
Top 3 by median price:
1. Runda - KSh 45M+
2. Karen - KSh 35M+
3. Kitisuru - KSh 30M+

### 2. **Size Impact on Price**
- Correlation: **0.65** (Strong positive relationship)
- Larger properties command significantly higher prices

### 3. **Amenity Impact**
- Properties with 3+ amenities cost **~40% more** than those with none
- Most valuable amenities: Parking, Security, Pool

---

## 🛠️ Technologies Used

- **Python 3.12**
- **Data Collection:** BeautifulSoup, Requests
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (upcoming)
- **Deployment:** Streamlit (upcoming)

---

##  How to Run

### 1. Clone the repository
```bash
git clone <https://github.com/mwandikikepha/Nairobi_Properties>
cd nairobi_property
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn requests beautifulsoup4
```

### 3. Run the scraper 
```bash
cd scrapping
python run_scrapper.py
```

### 4. Run the notebooks
```bash
jupyter notebook notebooks/day2_cleaning.ipynb
jupyter notebook notebooks/day2_eda.ipynb
```

---

##  Next Steps (Day 3)

- [ ] Train/test split
- [ ] Build Linear Regression baseline model
- [ ] Evaluate with MAE, RMSE, R²
- [ ] Interpret MAE in business terms (KES)

---



**Last Updated:** February 17, 2026