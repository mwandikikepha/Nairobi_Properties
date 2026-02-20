# Nairobi House Price Prediction


A complete machine learning pipeline for predicting residential property prices in Nairobi, Kenya. From web scraping to trained models, this project demonstrates end-to-end data science workflow for real estate price prediction.

---

##  Project Overview

This project predicts house prices in Nairobi based on property features like location, size, bedrooms, and amenities. The system includes:

- **Web scraper** for collecting property listings 
- **Data cleaning pipeline** removing outliers and standardizing formats
- **Machine Learning models** (Linear Regression & Random Forest)
- **Prediction scripts** for estimating new property prices
- **Visualization suite** for exploratory data analysis

### Key Results
- **clean property listings** collected and processed
- **Random Forest model** achieving R² = 0.801 (80.1% variance explained)
- **Mean Absolute Error** of ±4.93 Million KES
- **Production-ready** modular code and reusable functions

---

##  Problem Statement

Real estate pricing in Nairobi is opaque and inconsistent. Buyers, sellers, and agents lack data-driven tools to make informed decisions. This project addresses that gap by building a predictive model trained on real market data.

**Solution:** A machine learning model trained on 651 high-quality listings to predict prices within ±5M KES on average, helping stakeholders make evidence-based decisions.

---

##  Project Structure

```
nairobi_property/
├── data/
│   ├── raw_listings.csv          # Original scraped data (2,511 listings)
│   ├── clean_listings.csv        # Cleaned data (651 listings)
│   └── data_dictionary.md        # Data documentation
│
├── models/
│   └── model.pkl                 # Trained Random Forest model
│
├── notebooks/
│   ├── day1.ipynb                # Data collection & exploration
│   ├── day2_cleaning.ipynb       # Data cleaning process
│   ├── day2_eda.ipynb            # Exploratory data analysis
│   ├── day3_baseline_model.ipynb # Linear Regression baseline
│   └── day4_model_improvement.ipynb  # Random Forest training
│
├── scripts/
│   ├── clean_data.py             # Data cleaning functions
│   ├── feature_engineering.py    # Feature creation
│   ├── train_model.py            # Model training pipeline
│   ├── visualizations.py         # EDA plotting functions
│   └── predict.py                # Price prediction
│
├── scrapping/
│   ├── convert.py                # Data type conversions
│   ├── read_scrapped_data.py     # HTML parsing
│   ├── property_scraper.py       # Main scraper logic
│   └── run_scrapper.py           # Scraper execution
│
├── visualizations/               # Generated plots (PNG files)
│   ├── price_by_location.png
│   ├── size_vs_price.png
│   ├── price_by_bedrooms.png
│   └── price_per_sqft_by_location.png
│
├── app/
│   └── streamlit_app.py          # Web interface (future work)
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

##  Quick Start

### Prerequisites
- Python 3.12+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mwandikikepha/Nairobi_Properties.git
cd nairobi_property
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run predictions** (using pre-trained model)
```bash
cd scripts
python predict.py
```

---

##  Usage Examples

### 1. Make a Price Prediction

```python
from scripts.predict import load_model, prepare_input, predict_price, interpret_prediction

# Load trained model
model = load_model("models/model.pkl")

# Prepare property features
features = prepare_input(
    bedrooms=3,
    bathrooms=2,
    size_sqft=1500,
    location='Westlands',
    property_type='Apartment',
    amenity_score=2
)

# Get prediction
prediction = predict_price(model, features)
interpret_prediction(prediction, 3, 'Westlands', 'Apartment', 1500)
```

**Output:**
```

PRICE PREDICTION....

 3-bedroom Apartment in Westlands

 Predicted Price: KSh 18,500,000
   (18.50 Million KES)

 Estimated Range: KSh 13,570,000 - 23,430,000
   (Based on model MAE of ±4.93M)
```

### 2. Clean New Data

```bash
cd scripts
python clean_data.py
```

### 3. Generate Visualizations

```bash
cd scripts
python visualizations.py
```

### 4. Train Model from Scratch

```bash
cd scripts
python train_model.py
```

---

##  Data Pipeline

### 1. Data Collection
**Source:** Property24.co.ke (Nairobi residential listings)  
**Method:** Custom BeautifulSoup web scraper  
**Raw Data:** 2,511 listings scraped  
**Features Collected:** Location, property type, bedrooms, bathrooms, size (sqft), amenities, price (KES), listing date

### 2. Data Cleaning
**Cleaning Steps:**
- Removed  duplicate listings
- Filtered out  size outliers (<200 sqft or >20,000 sqft)
- Removed price outliers (<5M KES or >100M KES)
- Dropped listings with missing amenities
- Standardized location and property type names

**Result:** high-quality listings (74% data retention)

### 3. Feature Engineering
**Created Features:**
- `price_per_sqft`: Price normalized by size for comparison
- `amenity_score`: Count of amenities (0-5)
- `month`: Temporal feature extracted from listing date

### 4. Model Training
**Models Developed:**

| Model | MAE (M KES) | RMSE (M KES) | R² Score |
|-------|-------------|--------------|----------|
| Linear Regression | 7.33 | 11.23 | 0.607 |
| **Random Forest** | **4.94** | **11.14** | **0.801** |

**Winner:** Random Forest  
- 32% better MAE (7.33M → 4.94M KES)
- 27% better R² (0.607 → 0.801)
- Better handling of non-linear relationships

---

##  Key Insights

### Price Drivers (Feature Importance)
1. **Property Type** (45.7%) - Houses worth significantly more than apartments
2. **Size** (17.5%) - Larger properties command higher prices
3. **Bedrooms** (17.0%) - Each bedroom adds value
4. **Bathrooms** (9.4%) - Moderate impact
5. **Location** (9.1%) - Varies by neighborhood
6. **Amenities** (1.4%) - Minimal impact

### Most Expensive Locations (Median Price)
1. **Runda** - 53M KES
2. **Karen** - 35M KES
3. **Kitisuru** - 30M KES
4. **Lavington** - 28M KES
5. **Spring Valley** - 27M KES

### Market Statistics
- **Median Price:** 15.2M KES
- **Price Range:** 5M - 100M KES
- **Median Size:** 1,200 sqft
- **Median Price/sqft:** 10,926 KES

---

##  Technologies Used

**Core Stack:**
- Python 3.12
- Pandas & NumPy (Data manipulation)
- Scikit-learn (Machine learning)
- Matplotlib & Seaborn (Visualization)

**Data Collection:**
- BeautifulSoup4 (HTML parsing)
- Requests (HTTP requests)

**Development:**
- Jupyter Notebooks (Interactive analysis)
- Git (Version control)
- pickle (Model serialization)

---

##  Model Details

### Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

**Training Configuration:**
- Training Set: 80% (532 properties)
- Test Set: 20% (133 properties)
- Features: 6 (bedrooms, bathrooms, size_sqft, location, property_type, amenity_score)
- Target Variable: price_kes

**Performance:**
- MAE: 4.94M KES
- RMSE: 11.14M KES
- R²: 0.749

**Interpretation:** Model predictions are within ±4.94M KES of actual prices on average, explaining 74.9% of price variance.

---

##  Visualizations

The project includes publication-quality visualizations:

1. **Price Distribution by Location** - Boxplots for top 10 neighborhoods
2. **Size vs Price Scatter** - Correlation with trend line
3. **Price by Bedrooms** - Median and distribution analysis
4. **Price per Sqft by Location** - Cost efficiency comparison

All visualizations feature:
- Million KES formatting for readability
- Color gradients for visual appeal
- Grid lines for precise reading
- Professional styling

---


### Planned Improvements
- Expand to much listings from multiple sources
- Add GPS coordinates for better location features
- Implement time-series analysis for market trends
- Build interactive Streamlit web app
- Try XGBoost for potential accuracy gains
- Add property condition and age features

---

##  Use Cases

### For Home Buyers
- Evaluate if a property is fairly priced
- Compare prices across neighborhoods
- Budget effectively for desired locations

### For Sellers
- Set competitive listing prices based on data
- Identify which improvements add the most value
- Avoid overpricing or underpricing

### For Real Estate Agents
- Provide clients with data-driven estimates
- Quickly assess property values
- Identify undervalued investment opportunities

### For Property Developers
- Understand price trends across Nairobi
- Determine optimal property specifications
- Forecast returns on development projects

### For Researchers
- Analyze Nairobi real estate market trends
- Study feature-price relationships
- Generate policy insights

---

##  Contributing

Contributions, suggestions, and feedback are welcome!

**How to Contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---


##  Author

**Kepha Mwandiki**  
Data Engineer

- GitHub: [@mwandikikepha](https://github.com/mwandikikepha)
- Project: [Nairobi Properties](https://github.com/mwandikikepha/Nairobi_Properties)

---
