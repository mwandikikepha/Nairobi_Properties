"""
Visualization Script
Extracted from day2_eda.ipynb
Creates exploratory data analysis visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():

    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_price_by_location(df, save_path=None):
    
    top_10_locations = df['location'].value_counts().head(10).index
    df_top = df[df['location'].isin(top_10_locations)]
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df_top, x='location', y='price_kes', order=top_10_locations, palette='Set2')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.ylabel('Price (Millions KES)', fontsize=12, fontweight='bold')
    plt.xlabel('Location', fontsize=12, fontweight='bold')
    plt.title('Price Distribution by Location (Top 10)', fontsize=14, fontweight='bold', pad=20)
    
    # Format Y-axis to show millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print median prices
    median_by_location = df.groupby('location')['price_kes'].median().sort_values(ascending=False)
    print("\n Top 5 Most Expensive Locations (by median price):")
    for i, (loc, price) in enumerate(median_by_location.head().items(), 1):
        print(f"   {i}. {loc}: KSh {price/1e6:.1f}M")


def plot_size_vs_price(df, save_path=None):
    
    plt.figure(figsize=(12, 7))
    
    # Scatter plot with color gradient
    scatter = plt.scatter(df['size_sqft'], df['price_kes'], 
                         alpha=0.6, c=df['bedrooms'], cmap='viridis', 
                         edgecolors='black', linewidth=0.5, s=80)
    
    plt.colorbar(scatter, label='Bedrooms')
    plt.xlabel('Size (sqft)', fontsize=12, fontweight='bold')
    plt.ylabel('Price (Millions KES)', fontsize=12, fontweight='bold')
    plt.title('Price vs Size Relationship', fontsize=14, fontweight='bold', pad=20)
    
    # Format Y-axis to show millions
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add trend line
    z = np.polyfit(df['size_sqft'].dropna(), df['price_kes'][df['size_sqft'].notna()], 1)
    p = np.poly1d(z)
    plt.plot(df['size_sqft'], p(df['size_sqft']), "r--", alpha=0.8, linewidth=2, label='Trend')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_price_by_bedrooms(df, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Median price by bedrooms
    bedroom_median = df.groupby('bedrooms')['price_kes'].median() / 1e6
    bedroom_median.plot(kind='bar', color='#2ecc71', ax=axes[0], edgecolor='black')
    axes[0].set_xlabel('Number of Bedrooms', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Median Price (Millions KES)', fontsize=12, fontweight='bold')
    axes[0].set_title('Median Price by Bedrooms', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, v in enumerate(bedroom_median):
        axes[0].text(i, v + 1, f'{v:.1f}M', ha='center', fontweight='bold')
    
    # Boxplot
    sns.boxplot(data=df, x='bedrooms', y='price_kes', palette='Greens', ax=axes[1])
    axes[1].set_xlabel('Number of Bedrooms', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Price (Millions KES)', fontsize=12, fontweight='bold')
    axes[1].set_title('Price Distribution by Bedrooms', fontsize=13, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_price_per_sqft_by_location(df, save_path=None):
    
    top_locs = df['location'].value_counts().head(10).index
    df_top = df[df['location'].isin(top_locs)]
    
    plt.figure(figsize=(12, 7))
    price_per_sqft_by_loc = df_top.groupby('location')['price_per_sqft'].median().sort_values()
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(price_per_sqft_by_loc)))
    bars = price_per_sqft_by_loc.plot(kind='barh', color=colors, edgecolor='black')
    
    plt.xlabel('Median Price per sqft (KES)', fontsize=12, fontweight='bold')
    plt.ylabel('Location', fontsize=12, fontweight='bold')
    plt.title('Price per Square Foot by Location (Cheapest to Most Expensive)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(price_per_sqft_by_loc):
        plt.text(v + 300, i, f'KSh {v:,.0f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    
    # Select numerical columns
    numeric_cols = ['bedrooms', 'bathrooms', 'size_sqft', 'price_kes', 
                    'price_per_sqft', 'amenity_score']
    
    # Calculate correlations
    corr = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🔍 Strongest Correlations with Price:")
    price_corr = corr['price_kes'].drop('price_kes').sort_values(ascending=False)
    for feat, corr_val in price_corr.items():
        print(f"   {feat}: {corr_val:.2f}")

def plot_market_segments(df, save_path=None):
    
    # Create price categories
    df_segments = df.copy()
    df_segments['segment'] = pd.cut(
        df_segments['price_kes'],
        bins=[0, 10_000_000, 30_000_000, 100_000_000],
        labels=['Affordable (<10M)', 'Mid-Range (10-30M)', 
                'Premium (30-100M)']
    )
    
    segment_counts = df_segments['segment'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    # Pie chart with percentage
    wedges, texts, autotexts = plt.pie(
        segment_counts.values, 
        labels=segment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.05, 0.05, 0.05],
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    plt.title('Nairobi Property Market Segments', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with counts
    plt.legend([f'{label}: {count} properties' 
                for label, count in segment_counts.items()],
               loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Market Segment Breakdown:")
    for segment, count in segment_counts.items():
        percentage = (count / len(df_segments)) * 100
        print(f"   {segment}: {count} properties ({percentage:.1f}%)")

def plot_property_type_distribution(df, save_path=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    type_counts = df['property_type'].value_counts()
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    axes[0].bar(type_counts.index, type_counts.values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Property Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Properties by Type', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(type_counts.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    # Price by type
    type_price = df.groupby('property_type')['price_kes'].median() / 1e6
    axes[1].bar(type_price.index, type_price.values, color=colors, edgecolor='black')
    axes[1].set_xlabel('Property Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Median Price (Millions KES)', fontsize=12, fontweight='bold')
    axes[1].set_title('Median Price by Type', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(type_price.values):
        axes[1].text(i, v + 1, f'{v:.1f}M', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🏠 Property Type Insights:")
    for prop_type in df['property_type'].unique():
        count = len(df[df['property_type'] == prop_type])
        avg_price = df[df['property_type'] == prop_type]['price_kes'].mean() / 1e6
        print(f"   {prop_type}: {count} properties, Avg Price: {avg_price:.1f}M KES")


def print_data_quality_check(df):
    
    
    print("\n Price distribution:")
    print(df['price_kes'].describe())
    
    print(f"\nProperties > 100M: {len(df[df['price_kes'] > 100_000_000])}")
    print(f"Missing size_sqft: {df['size_sqft'].isna().sum()}")
    
    print("\nPrice per sqft outliers:")
    print(df['price_per_sqft'].describe())


def create_all_visualizations(input_path, save_dir=None):
    
    # Setup
    setup_plot_style()
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} clean listings for visualization")
    
    # Create visualizations
    print("\n Creating visualizations...")
    
    # Original plots
    print("\n1. Price by Location")
    plot_price_by_location(df, f"{save_dir}/price_by_location.png" if save_dir else None)
    
    print("\n2. Size vs Price")
    plot_size_vs_price(df, f"{save_dir}/size_vs_price.png" if save_dir else None)
    
    print("\n3. Price by Bedrooms")
    plot_price_by_bedrooms(df, f"{save_dir}/price_by_bedrooms.png" if save_dir else None)
    
    print("\n4. Price per sqft by Location")
    plot_price_per_sqft_by_location(df, f"{save_dir}/price_per_sqft_by_location.png" if save_dir else None)
    
    # NEW PLOTS
    print("\n5. Correlation Heatmap")
    plot_correlation_heatmap(df, f"{save_dir}/correlation_heatmap.png" if save_dir else None)
    
    print("\n6. Market Segments")
    plot_market_segments(df, f"{save_dir}/market_segments.png" if save_dir else None)
    
    print("\n7. Property Type Distribution")
    plot_property_type_distribution(df, f"{save_dir}/property_types.png" if save_dir else None)
    
    # Data quality check
    print_data_quality_check(df)
    
    print("\n All visualizations created!")
    
    return df


if __name__ == "__main__":
    # Run all visualizations
    input_file = "../data/clean_listings.csv"
    
    
    df = create_all_visualizations(input_file, save_dir="../visualizations")
    
    print(f"\n Visualizations complete for {len(df)} properties")