"""
Nairobi Property Price Predictor
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Nairobi Property Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(" Model file not found. Please ensure model.pkl exists in models/ directory.")
        return None


@st.cache_data
def load_data():
    """Load clean listings data"""
    try:
        df = pd.read_csv('data/clean_listings.csv')
        return df
    except FileNotFoundError:
        st.error(" Data file not found. Please ensure clean_listings.csv exists in data/ directory.")
        return None


def prepare_encoders(df):
    """Prepare label encoders from data"""
    le_location = LabelEncoder()
    le_location.fit(df['location'])
    
    le_property = LabelEncoder()
    le_property.fit(df['property_type'])
    
    return le_location, le_property


def predict_price(model, bedrooms, bathrooms, size_sqft, location_encoded, property_encoded, amenity_score):
    """Make price prediction"""
    features = np.array([[bedrooms, bathrooms, size_sqft, location_encoded, property_encoded, amenity_score]])
    prediction = model.predict(features)[0]
    return prediction


def main():
    # Header
    st.markdown('<p class="main-header">🏠 Nairobi Property Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get instant AI-powered price estimates for properties in Nairobi</p>', unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    df = load_data()
    
    if model is None or df is None:
        st.stop()
    
    # Prepare encoders
    le_location, le_property = prepare_encoders(df)
    
    # Sidebar - Input Form
    st.sidebar.header(" Property Details")
    st.sidebar.markdown("Enter the property specifications below:")
    
    # Property Type
    property_type = st.sidebar.selectbox(
        "Property Type",
        options=sorted(df['property_type'].unique()),
        help="Select the type of property"
    )
    
    # Location
    location = st.sidebar.selectbox(
        "Location",
        options=sorted(df['location'].unique()),
        help="Select the neighborhood in Nairobi"
    )
    
    # Bedrooms
    bedrooms = st.sidebar.slider(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of bedrooms in the property"
    )
    
    # Bathrooms
    bathrooms = st.sidebar.slider(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of bathrooms in the property"
    )
    
    # Size
    size_sqft = st.sidebar.number_input(
        "Size (Square Feet)",
        min_value=200,
        max_value=10000,
        value=1500,
        step=100,
        help="Property size in square feet"
    )
    
    # Amenities
    amenity_score = st.sidebar.slider(
        "Number of Amenities",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of amenities (parking, pool, gym, security, garden)"
    )
    
    # Predict button
    predict_button = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Encode inputs
            location_encoded = le_location.transform([location])[0]
            property_encoded = le_property.transform([property_type])[0]
            
            # Make prediction
            prediction = predict_price(
                model, bedrooms, bathrooms, size_sqft, 
                location_encoded, property_encoded, amenity_score
            )
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2> Predicted Price</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">
                    KSh {prediction:,.0f}
                </h1>
                <h3>{prediction/1_000_000:.2f} Million KES</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Price range (based on MAE)
            mae = 4_940_000
            lower = max(0, prediction - mae)
            upper = prediction + mae
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Lower Estimate", f"KSh {lower:,.0f}")
                st.caption(f"{lower/1_000_000:.2f}M KES")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Expected Price", f"KSh {prediction:,.0f}")
                st.caption(f"{prediction/1_000_000:.2f}M KES")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Upper Estimate", f"KSh {upper:,.0f}")
                st.caption(f"{upper/1_000_000:.2f}M KES")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Price per sqft
            price_per_sqft = prediction / size_sqft
            st.info(f" **Price per Square Foot:** KSh {price_per_sqft:,.0f}")
            
            # Property summary
            st.markdown("---")
            st.subheader(" Property Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write(f"**Type:** {property_type}")
                st.write(f"**Location:** {location}")
                st.write(f"**Bedrooms:** {bedrooms}")
            
            with summary_col2:
                st.write(f"**Bathrooms:** {bathrooms}")
                st.write(f"**Size:** {size_sqft:,} sqft")
                st.write(f"**Amenities:** {amenity_score}")
            
        else:
            # Welcome message
            st.markdown("""
            ##  Welcome to the Nairobi Property Price Predictor
            
            This intelligent tool uses **machine learning** to provide accurate property price estimates in Nairobi. 
            Our AI model analyzes multiple factors to give you reliable predictions.
            
            ###  What We Consider
            - **Location** - Prime neighborhoods across Nairobi
            - **Property Type** - Apartments, Houses, Townhouses, and Villas
            - **Size** - Total square footage of the property
            - **Bedrooms & Bathrooms** - Room configuration
            - **Amenities** - Features like parking, pool, gym, security, and garden
            
            ###  Why Use This Tool?
            - **Fast & Accurate** - Get instant price estimates backed by real market data
            - **Data-Driven** - Predictions based on actual Nairobi property transactions
            - **Easy to Use** - Simple interface with intuitive controls
            - **Comprehensive** - Covers major residential areas in Nairobi
            
            ###  How to Get Started
            1. Select your property specifications in the sidebar
            2. Click the "Predict Price" button
            3. Get your estimated price range instantly
            
            ---
            
            **Note:** Predictions are estimates based on market trends. Actual prices may vary depending on property condition, exact location, and current market dynamics.
            """)
    
    with col2:
        st.subheader(" Market Insights")
        
        # Most expensive locations
        st.markdown("#### Top 5 Expensive Areas")
        top_locations = df.groupby('location')['price_kes'].median().sort_values(ascending=False).head(5)
        
        for i, (loc, price) in enumerate(top_locations.items(), 1):
            st.write(f"{i}. **{loc}** - {price/1_000_000:.1f}M KES")
        
        st.markdown("---")
        
        # Price by bedrooms
        st.markdown("#### Average Price by Bedrooms")
        bedroom_prices = df.groupby('bedrooms')['price_kes'].median() / 1_000_000
        
        fig = go.Figure(data=[
            go.Bar(
                x=bedroom_prices.index,
                y=bedroom_prices.values,
                marker_color='lightblue',
                text=[f'{v:.1f}M' for v in bedroom_prices.values],
                textposition='auto',
            )
        ])
        fig.update_layout(
            xaxis_title="Bedrooms",
            yaxis_title="Median Price (M KES)",
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Dataset info (general)
        st.markdown("####  Market Coverage")
        st.write(f"**Neighborhoods:** {df['location'].nunique()} locations")
        st.write(f"**Property Types:** {df['property_type'].nunique()} types")
        st.write(f"**Price Range:** {df['price_kes'].min()/1_000_000:.1f}M - {df['price_kes'].max()/1_000_000:.1f}M KES")
        st.write(f"**Median Price:** {df['price_kes'].median()/1_000_000:.1f}M KES")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built using Python, Scikit-learn, and Streamlit</p>
        <p>Data Source: Property24.co.ke | Powered by Advanced Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()