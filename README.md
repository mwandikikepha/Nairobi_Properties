# Nairobi House Price Prediction

**6-Day Intensive Sprint - Prop-Tech MVP**

A machine learning project to predict house prices in Nairobi, Kenya using data from Nairobi Propery Listings.

---

##  Project Status

**Current Phase:** Day 3 Complete 

-  Day 1: Data Collection & Structuring (801 listings scraped)
-  Day 2: Data Cleaning & Feature Engineering (326 clean listings)
-  Day 3: Baseline Model - Linear Regression (R²=0.745)
-  Day 4: Model Improvement & Explainability
-  Day 5: Build Pricing App (Deployment)
-  Day 6: Dashboard & Final Presentation

---

##  Project Structure
```
nairobi_property/
├── data/
│   ├── raw_listings.csv           # Original scraped data (801 listings)
│   ├── clean_listings.csv         # Cleaned data (326 listings)
│   └── data_dictionary.md         # Data documentation
├── notebooks/
│   ├── day1.ipynb                 # Data collection notebook
│   ├── day2_cleaning.ipynb        # Data cleaning & feature engineering
│   ├── day2_eda.ipynb             # Exploratory data analysis
│   └── day3_baseline_model.ipynb  # Baseline Linear Regression model
├── scrapping/
│   ├── convert.py                 # Data conversion utilities
│   ├── read_scrapped_data.py      # HTML parser
│   ├── property_scraper.py        # Main scraper
│   └── run_scrapper.py            # Scraper runner
└── README.md
```

---

##  Dataset Overview

**Source:** Web Scrapping 
**Collection Date:** February 16, 2026  
**Raw Data:** 801 listings  
**Clean Data:** 326 listings (after removing outliers)

### Key Features:
- **Location** - Nairobi neighborhoods (18 unique areas)
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

##  Model Performance (Day 3 Baseline)

**Algorithm:** Linear Regression  
**Training Data:** 260 samples (80%)  
**Test Data:** 65 samples (20%)

### Metrics:
- **MAE:** ±9.79 Million KES
- **RMSE:** 16.25 Million KES  
- **R²:** 0.745 (explains 74.5% of price variance)

### Key Findings:
-  Property type is strongest predictor (+27.7M for Houses/Villas)
-  Each bedroom adds ~14.6M KES
-  Each sqft adds ~5,348 KES
-  Model performs well on properties <50M KES
-  Struggles with luxury properties (>100M KES) - to be improved in Day 4

---

##  Technologies Used

- **Python 3.12**
- **Data Collection:** BeautifulSoup, Requests
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (Linear Regression, upcoming: Random Forest, XGBoost)
- **Deployment:** Streamlit (upcoming)

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/mwandikikepha/Nairobi_Properties
cd nairobi_property
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn requests beautifulsoup4 scikit-learn
```

### 3. Run the scraper (optional)
```bash
cd scrapping
python run_scrapper.py
```

### 4. Run the notebooks
```bash
jupyter notebook notebooks/day2_cleaning.ipynb
jupyter notebook notebooks/day2_eda.ipynb
jupyter notebook notebooks/day3_baseline_model.ipynb
```

---

##  Next Steps (Day 4)

- [ ] Train Random Forest model
- [ ] Train XGBoost model (if time allows)
- [ ] Compare model performance
- [ ] Extract feature importance
- [ ] Improve predictions on luxury properties

---


**Last Updated:** February 18, 2026