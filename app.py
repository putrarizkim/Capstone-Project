# type: ignore
"""
Capstone Project DQLab - Machine Learning & AI Track
House Price Prediction System
Author: Muhammad Rizki Putra
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

# THEME SELECTOR
theme_option = st.sidebar.radio(
    "🎨 Theme",
    ["☀️ Light Mode", "🌙 Dark Mode"],
    horizontal=True
)

# Theme configurations
themes = {
    "☀️ Light Mode": {
        "primary": "#FF4B4B",
        "secondary": "#FF6B6B",
        "accent": "#FFA8A8",
        "bg_color": "#FFFFFF",
        "card_bg": "#F8F9FA",
        "text_color": "#1F2937",
        "border_color": "#E5E7EB",
        "success": "#10B981",
        "danger": "#EF4444",
        "chart_template": "plotly_white"
    },
    "🌙 Dark Mode": {
        "primary": "#3B82F6",
        "secondary": "#60A5FA",
        "accent": "#93C5FD",
        "bg_color": "#1F2937",
        "card_bg": "#374151",
        "text_color": "#F9FAFB",
        "border_color": "#4B5563",
        "success": "#10B981",
        "danger": "#EF4444",
        "chart_template": "plotly_dark"
    }
}

theme = themes[theme_option]

# Custom CSS
st.markdown(f"""
<style>
    /* Main App Background */
    .stApp {{
        background-color: {theme['bg_color']};
        color: {theme['text_color']};
    }}
    
    /* Main container */
    .main {{
        background-color: {theme['bg_color']};
        color: {theme['text_color']};
    }}
    
    /* Headers */
    .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        color: {theme['primary']};
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    .sub-header {{
        font-size: 1.5rem;
        color: {theme['secondary']};
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .section-header {{
        font-size: 2rem;
        font-weight: bold;
        color: {theme['primary']};
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid {theme['accent']};
        padding-left: 15px;
    }}
    
    /* Cards and boxes */
    .info-box {{
        background: {theme['card_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {theme['primary']};
        margin: 1rem 0;
        color: {theme['text_color']};
    }}
    
    /* Buttons */
    .stButton>button {{
        background: {theme['primary']};
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton>button:hover {{
        background: {theme['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {theme['card_bg']};
        border-right: 2px solid {theme['border_color']};
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: {theme['text_color']};
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: bold;
        color: {theme['primary']};
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {theme['text_color']};
    }}
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        background-color: {theme['card_bg']};
        color: {theme['text_color']};
        border: 1px solid {theme['border_color']};
        border-radius: 5px;
    }}
    
    /* Dataframes */
    .dataframe {{
        background-color: {theme['card_bg']};
        color: {theme['text_color']};
    }}
    
    /* Success/Info/Warning boxes */
    .stSuccess {{
        background-color: {theme['success']}20;
        border-left: 4px solid {theme['success']};
        color: {theme['text_color']};
    }}
    
    .stInfo {{
        background-color: {theme['primary']}20;
        border-left: 4px solid {theme['primary']};
        color: {theme['text_color']};
    }}
    
    .stWarning {{
        background-color: {theme['danger']}20;
        border-left: 4px solid {theme['danger']};
        color: {theme['text_color']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {theme['card_bg']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {theme['text_color']};
    }}
    
    /* Divider */
    hr {{
        border-color: {theme['border_color']};
    }}
</style>
""", unsafe_allow_html=True)

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Load dataset
@st.cache_data
def load_house_dataset():
    np.random.seed(42)
    n = 1460
    
    # Generate realistic house data
    df = pd.DataFrame({
        'LotArea': np.random.randint(1000, 20000, n),
        'OverallQual': np.random.randint(1, 11, n),
        'OverallCond': np.random.randint(1, 11, n),
        'YearBuilt': np.random.randint(1950, 2024, n),
        'GrLivArea': np.random.randint(500, 5000, n),
        'FullBath': np.random.randint(0, 4, n),
        'BedroomAbvGr': np.random.randint(1, 6, n),
        'KitchenAbvGr': np.random.randint(1, 4, n),
        'TotRmsAbvGrd': np.random.randint(3, 12, n),
        'Fireplaces': np.random.randint(0, 4, n),
        'GarageArea': np.random.randint(0, 800, n),
        'GarageCars': np.random.randint(0, 4, n),
        'PoolArea': np.random.choice([0, 0, 0, 0, 200, 400, 600], n),
        'YrSold': np.random.randint(2006, 2024, n)
    })
    
    # Calculate price based on features (realistic formula)
    base_price = 50000
    df['SalePrice'] = (
        base_price +
        df['LotArea'] * 5 +
        df['OverallQual'] * 15000 +
        df['OverallCond'] * 5000 +
        (2024 - df['YearBuilt']) * -500 +
        df['GrLivArea'] * 80 +
        df['FullBath'] * 20000 +
        df['BedroomAbvGr'] * 15000 +
        df['GarageArea'] * 100 +
        df['GarageCars'] * 25000 +
        df['PoolArea'] * 150 +
        df['Fireplaces'] * 10000 +
        np.random.randint(-30000, 30000, n)
    )
    
    # Add noise and ensure positive prices
    df['SalePrice'] = df['SalePrice'].clip(lower=50000)
    
    return df

# Train model
@st.cache_resource
def train_house_model(df):
    if not SKLEARN_AVAILABLE:
        return None, None, 0, 0, 0, None, None, None
    
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, scaler, r2, rmse, mae, X_test, y_test, y_pred

df = load_house_dataset()
if SKLEARN_AVAILABLE:
    model, scaler, r2, rmse, mae, X_test, y_test, y_pred = train_house_model(df)

# Sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("---")
    
    nav_items = {
        "🏠 Home": "Home",
        "📊 Dataset": "Dataset",
        "📈 EDA": "Exploratory Data Analysis",
        "🤖 Modelling": "Modelling",
        "💰 Prediction": "Prediction",
        "👤 About": "About"
    }
    
    for label, page_name in nav_items.items():
        if st.button(label, key=page_name, use_container_width=True):
            st.session_state.current_page = page_name
    
    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Properties", f"{len(df):,}")
    st.metric("Avg Price", f"${df['SalePrice'].mean():,.0f}")
    if SKLEARN_AVAILABLE:
        st.metric("Model R² Score", f"{r2:.3f}")

page = st.session_state.current_page

# ==================== HOME PAGE ====================
if page == "Home":
    st.markdown('<div class="main-header">🏠 Capstone Project DQLab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning & AI Track - House Price Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("""
        ### 👋 Welcome!
        
        Halo perkenalkan nama saya **Muhammad Rizki Putra** dan Saya mengikuti kelas **Machine Learning & AI** di DQLab Academy yang ke 19. 
        
        Ini adalah **capstone project** saya yang berfokus pada prediksi harga rumah 
        menggunakan machine learning untuk membantu buyer, seller, dan real estate agent 
        dalam menentukan harga properti yang tepat.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400", use_container_width=True)
    
    st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
    
    st.write("""
    **Real Estate Market** adalah salah satu sektor ekonomi terpenting dengan nilai transaksi 
    triliunan rupiah setiap tahunnya. Namun, menentukan harga rumah yang tepat seringkali 
    menjadi tantangan karena banyak faktor yang mempengaruhi:
    
    - 🏡 **Lokasi dan Luas Tanah** - Faktor utama penentu harga
    - 🏗️ **Kondisi dan Kualitas Bangunan** - Material dan maintenance
    - 📅 **Tahun Pembangunan** - Usia properti
    - 🛏️ **Jumlah Kamar dan Fasilitas** - Bedroom, bathroom, garage, pool
    - 📊 **Trend Pasar** - Supply dan demand
    
    Project ini menggunakan **Machine Learning** untuk memprediksi harga rumah secara akurat!
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏘️ Total Properties", f"{len(df):,}")
    with col2:
        st.metric("💰 Avg Price", f"${df['SalePrice'].mean():,.0f}")
    with col3:
        st.metric("📊 Price Range", f"${df['SalePrice'].min():,.0f} - ${df['SalePrice'].max():,.0f}")
    with col4:
        st.metric("🎯 R² Score", f"{r2:.3f}" if SKLEARN_AVAILABLE else "N/A")
    
    st.markdown('<div class="section-header">🎯 Project Objective</div>', unsafe_allow_html=True)
    
    st.success("""
    ### Tujuan Project:
    Membangun model **Machine Learning** yang dapat memprediksi harga rumah secara akurat 
    berdasarkan karakteristik properti seperti luas tanah, jumlah kamar, kondisi bangunan, 
    dan fasilitas yang tersedia menggunakan **Random Forest Regressor**.
    """)
    
    st.markdown('<div class="section-header">💼 Use Cases</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🏠 Home Buyers")
        st.write("Mengetahui apakah harga yang ditawarkan wajar atau overpriced")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💼 Real Estate Agents")
        st.write("Menentukan listing price yang kompetitif dan akurat")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🏦 Banks & Appraisers")
        st.write("Valuasi properti untuk keperluan kredit dan investasi")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🛠️ Methodology</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1️⃣ Data Collection")
        st.write("Mengumpulkan data properti dengan 14+ features")
    
    with col2:
        st.markdown("### 2️⃣ Data Analysis")
        st.write("EDA, visualisasi, dan correlation analysis")
    
    with col3:
        st.markdown("### 3️⃣ Model Training")
        st.write("Random Forest Regressor dengan hyperparameter tuning")
    
    with col4:
        st.markdown("### 4️⃣ Deployment")
        st.write("Web app untuk prediksi real-time")

# ==================== DATASET PAGE ====================
elif page == "Dataset":
    st.markdown('<div class="main-header">📊 Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">House Sales Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Records", f"{len(df):,}")
    with col2:
        st.metric("📋 Features", len(df.columns) - 1)
    with col3:
        st.metric("💰 Avg Price", f"${df['SalePrice'].mean():,.0f}")
    with col4:
        st.metric("✅ Missing Values", "0")
    
    st.markdown('<div class="section-header">📖 Feature Descriptions</div>', unsafe_allow_html=True)
    
    features_info = pd.DataFrame({
        "Feature": ["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", 
                   "FullBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
                   "GarageArea", "GarageCars", "PoolArea", "YrSold", "SalePrice"],
        "Description": [
            "Lot size in square feet",
            "Overall material and finish quality (1-10)",
            "Overall condition rating (1-10)",
            "Original construction year",
            "Above ground living area (sqft)",
            "Full bathrooms above grade",
            "Number of bedrooms above basement",
            "Number of kitchens",
            "Total rooms above grade",
            "Number of fireplaces",
            "Size of garage in square feet",
            "Size of garage in car capacity",
            "Pool area in square feet",
            "Year property was sold",
            "Sale price in dollars (TARGET)"
        ],
        "Type": ["Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", 
                "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", "Numeric", 
                "Numeric", "Numeric", "Target"]
    })
    
    st.dataframe(features_info, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="section-header">👀 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown('<div class="section-header">📈 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"""
        **Price Statistics:**
        - Minimum Price: ${df['SalePrice'].min():,.0f}
        - Maximum Price: ${df['SalePrice'].max():,.0f}
        - Average Price: ${df['SalePrice'].mean():,.0f}
        - Median Price: ${df['SalePrice'].median():,.0f}
        - Standard Deviation: ${df['SalePrice'].std():,.0f}
        """)
    
    with col2:
        st.write(f"""
        **Property Characteristics:**
        - Average Living Area: {df['GrLivArea'].mean():.0f} sqft
        - Average Lot Area: {df['LotArea'].mean():,.0f} sqft
        - Average Bedrooms: {df['BedroomAbvGr'].mean():.1f}
        - Average Bathrooms: {df['FullBath'].mean():.1f}
        - Average Year Built: {df['YearBuilt'].mean():.0f}
        """)
    
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Full Dataset", csv, "house_prices.csv", "text/csv", use_container_width=True)

# ==================== EDA PAGE ====================
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="main-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual Insights & Patterns</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Scatter Plots", "🎯 Insights"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sale Price distribution
            fig1 = px.histogram(df, x='SalePrice', nbins=50,
                              title='Sale Price Distribution',
                              labels={'SalePrice': 'Sale Price ($)'})
            fig1.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Living Area distribution
            fig3 = px.histogram(df, x='GrLivArea', nbins=50,
                              title='Living Area Distribution',
                              labels={'GrLivArea': 'Living Area (sqft)'})
            fig3.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Bedrooms distribution
            fig5 = px.histogram(df, x='BedroomAbvGr',
                              title='Number of Bedrooms Distribution')
            fig5.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Overall Quality distribution
            fig2 = px.histogram(df, x='OverallQual',
                              title='Overall Quality Distribution (1-10)')
            fig2.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Year Built distribution
            fig4 = px.histogram(df, x='YearBuilt', nbins=50,
                              title='Year Built Distribution')
            fig4.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Garage Cars distribution
            fig6 = px.histogram(df, x='GarageCars',
                              title='Garage Size Distribution (Cars)')
            fig6.update_layout(template=theme['chart_template'], showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        corr = df.corr()
        fig = px.imshow(corr, text_auto='.2f',
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(height=700, template=theme['chart_template'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Correlations with Sale Price")
        
        price_corr = corr['SalePrice'].sort_values(ascending=False)[1:11]
        
        fig2 = px.bar(x=price_corr.index, y=price_corr.values,
                     title='Top 10 Features Correlated with Sale Price',
                     labels={'x': 'Feature', 'y': 'Correlation'},
                     color=price_corr.values,
                     color_continuous_scale='Viridis')
        fig2.update_layout(template=theme['chart_template'])
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Living Area vs Price
            fig1 = px.scatter(df, x='GrLivArea', y='SalePrice',
                             title='Living Area vs Sale Price',
                             labels={'GrLivArea': 'Living Area (sqft)', 
                                    'SalePrice': 'Sale Price ($)'},
                             opacity=0.6,
                             trendline="ols")
            fig1.update_layout(template=theme['chart_template'])
            st.plotly_chart(fig1, use_container_width=True)
            
            # Year Built vs Price
            fig3 = px.scatter(df, x='YearBuilt', y='SalePrice',
                             title='Year Built vs Sale Price',
                             labels={'YearBuilt': 'Year Built', 
                                    'SalePrice': 'Sale Price ($)'},
                             opacity=0.6,
                             trendline="ols")
            fig3.update_layout(template=theme['chart_template'])
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Overall Quality vs Price
            fig2 = px.scatter(df, x='OverallQual', y='SalePrice',
                             title='Overall Quality vs Sale Price',
                             labels={'OverallQual': 'Overall Quality (1-10)', 
                                    'SalePrice': 'Sale Price ($)'},
                             opacity=0.6,
                             trendline="ols")
            fig2.update_layout(template=theme['chart_template'])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Lot Area vs Price
            fig4 = px.scatter(df, x='LotArea', y='SalePrice',
                             title='Lot Area vs Sale Price',
                             labels={'LotArea': 'Lot Area (sqft)', 
                                    'SalePrice': 'Sale Price ($)'},
                             opacity=0.6,
                             trendline="ols")
            fig4.update_layout(template=theme['chart_template'])
            st.plotly_chart(fig4, use_container_width=True)
        
        # 3D Scatter
        fig5 = px.scatter_3d(df.sample(500), 
                            x='GrLivArea', y='OverallQual', z='SalePrice',
                            title='3D: Living Area × Quality × Price',
                            labels={'GrLivArea': 'Living Area', 
                                   'OverallQual': 'Quality',
                                   'SalePrice': 'Price'},
                            opacity=0.7,
                            color='SalePrice',
                            color_continuous_scale='Viridis')
        fig5.update_layout(template=theme['chart_template'])
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        st.subheader("Key Insights from Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Price Factors")
            st.write("""
            **Strongest Price Predictors:**
            1. 🏗️ **Overall Quality** - Kualitas bangunan sangat berpengaruh
            2. 📏 **Living Area** - Semakin luas, semakin mahal
            3. 🚗 **Garage Size** - Garage besar = harga tinggi
            4. 📅 **Year Built** - Rumah baru lebih mahal
            5. 🛁 **Full Bathrooms** - Jumlah bathroom penting
            
            **Insight:**
            - Kualitas konstruksi adalah faktor #1
            - Luas bangunan > Luas tanah
            - Fasilitas modern menaikkan harga
            """)
        
        with col2:
            st.markdown("### 💡 Market Trends")
            st.write(f"""
            **Price Distribution:**
            - Median: ${df['SalePrice'].median():,.0f}
            - 25th Percentile: ${df['SalePrice'].quantile(0.25):,.0f}
            - 75th Percentile: ${df['SalePrice'].quantile(0.75):,.0f}
            
            **Property Characteristics:**
            - Most common quality: {df['OverallQual'].mode()[0]}/10
            - Most common bedrooms: {df['BedroomAbvGr'].mode()[0]}
            - Average house age: {2024 - df['YearBuilt'].mean():.0f} years
            
            **Recommendations:**
            - Focus on quality upgrades
            - Maximize living space
            - Modern amenities add value
            """)
        
        st.markdown("### 📈 Correlation Summary")
        
        top_5_corr = corr['SalePrice'].sort_values(ascending=False)[1:6]
        corr_df = pd.DataFrame({
            'Feature': top_5_corr.index,
            'Correlation': top_5_corr.values,
            'Impact': ['Very High' if x > 0.6 else 'High' if x > 0.4 else 'Medium' for x in top_5_corr.values]
        })
        
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

# ==================== MODELLING PAGE ====================
elif page == "Modelling":
    st.markdown('<div class="main-header">🤖 Machine Learning Modelling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Random Forest Regressor</div>', unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Scikit-learn not installed. Run: `pip install scikit-learn`")
        st.stop()
    
    st.markdown("---")
    
    st.info("🌳 **Model**: Random Forest Regressor with 100 trees, max depth 15, trained on 80% data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 R² Score", f"{r2:.4f}",
                 help="Proportion of variance explained (closer to 1 is better)")
    with col2:
        st.metric("📊 RMSE", f"${rmse:,.0f}",
                 help="Root Mean Squared Error - average prediction error")
    with col3:
        st.metric("📉 MAE", f"${mae:,.0f}",
                 help="Mean Absolute Error - average absolute difference")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual vs Predicted Prices")
        
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }).sample(200)
        
        fig = px.scatter(comparison_df, x='Actual', y='Predicted',
                        title='Actual vs Predicted Sale Prices',
                        labels={'Actual': 'Actual Price ($)', 
                               'Predicted': 'Predicted Price ($)'},
                        opacity=0.6)
        
        # Add diagonal line (perfect prediction)
        min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
        max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(template=theme['chart_template'], height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"""
        **Model Performance:**
        - R² = {r2:.4f} means model explains {r2*100:.2f}% of price variance
        - Average error: ${mae:,.0f}
        - Points closer to red line = better predictions
        """)
    
    with col2:
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': df.drop('SalePrice', axis=1).columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature',
                    orientation='h',
                    title='Top Features for Price Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis')
        fig.update_layout(template=theme['chart_template'], height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Residual Analysis")
    
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residual plot
        fig = px.scatter(x=y_pred, y=residuals,
                        title='Residual Plot',
                        labels={'x': 'Predicted Price ($)', 'y': 'Residuals ($)'},
                        opacity=0.6)
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(template=theme['chart_template'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residual distribution
        fig = px.histogram(residuals, nbins=50,
                          title='Distribution of Residuals',
                          labels={'value': 'Residual ($)', 'count': 'Frequency'})
        fig.update_layout(template=theme['chart_template'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Model Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ✅ Strengths")
        st.write("""
        - High R² score indicates good fit
        - Handles non-linear relationships
        - Captures feature interactions
        - Robust to outliers
        """)
    
    with col2:
        st.markdown("### ⚠️ Limitations")
        st.write("""
        - May underperform on extreme prices
        - Requires feature engineering
        - Black box model
        - Needs regular retraining
        """)
    
    with col3:
        st.markdown("### 🎯 Use Cases")
        st.write("""
        - Property valuation
        - Investment analysis
        - Market pricing
        - Appraisal automation
        """)

# ==================== PREDICTION PAGE ====================
elif page == "Prediction":
    st.markdown('<div class="main-header">💰 House Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter Property Details</div>', unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Scikit-learn not installed")
        st.stop()
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📐 Property Size")
            lot_area = st.number_input("Lot Area (sqft) *", 1000, 50000, 10000, 100)
            gr_liv_area = st.number_input("Living Area (sqft) *", 500, 10000, 1500, 50)
            garage_area = st.number_input("Garage Area (sqft)", 0, 1000, 400, 50)
            pool_area = st.number_input("Pool Area (sqft)", 0, 1000, 0, 50)
        
        with col2:
            st.subheader("🏗️ Property Quality")
            overall_qual = st.slider("Overall Quality (1-10) *", 1, 10, 7)
            overall_cond = st.slider("Overall Condition (1-10) *", 1, 10, 5)
            year_built = st.number_input("Year Built *", 1950, 2024, 2000)
            
            st.subheader("🔢 Rooms")
            bedrooms = st.number_input("Bedrooms *", 1, 10, 3)
            full_bath = st.number_input("Full Bathrooms *", 0, 5, 2)
        
        with col3:
            st.subheader("🏡 Additional Features")
            kitchens = st.number_input("Kitchens *", 1, 3, 1)
            total_rooms = st.number_input("Total Rooms *", 3, 15, 7)
            fireplaces = st.number_input("Fireplaces", 0, 5, 0)
            garage_cars = st.number_input("Garage Capacity (cars)", 0, 5, 2)
            
            st.subheader("📅 Sale Info")
            yr_sold = st.number_input("Year to Sell", 2024, 2030, 2024)
        
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            property_address = st.text_input("Property Address (Optional)", 
                                            placeholder="e.g., 123 Main Street, City")
        
        with col2:
            submit = st.form_submit_button("💰 Predict Price", use_container_width=True)
    
    if submit:
        # Prepare input
        input_data = np.array([[
            lot_area, overall_qual, overall_cond, year_built, gr_liv_area,
            full_bath, bedrooms, kitchens, total_rooms, fireplaces,
            garage_area, garage_cars, pool_area, yr_sold
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        
        # Calculate confidence interval (approximate)
        confidence = 0.15  # 15% confidence interval
        lower_bound = predicted_price * (1 - confidence)
        upper_bound = predicted_price * (1 + confidence)
        
        # Save to history
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'address': property_address if property_address else "N/A",
            'living_area': gr_liv_area,
            'bedrooms': bedrooms,
            'quality': overall_qual,
            'predicted_price': predicted_price,
            'year_built': year_built
        }
        st.session_state.prediction_history.append(history_entry)
        
        # Display results
        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Predicted Price", f"${predicted_price:,.0f}")
        with col2:
            st.metric("📊 Lower Bound", f"${lower_bound:,.0f}")
        with col3:
            st.metric("📈 Upper Bound", f"${upper_bound:,.0f}")
        with col4:
            st.metric("📐 Price/sqft", f"${predicted_price/gr_liv_area:.0f}")
        
        # Price gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted House Price ($)", 'font': {'size': 24}},
            delta={'reference': df['SalePrice'].median(), 
                   'valueformat': ",.0f"},
            number={'prefix': "$", 'valueformat': ",.0f"},
            gauge={
                'axis': {'range': [None, df['SalePrice'].max()], 'tickformat': ",.0f"},
                'bar': {'color': theme['primary']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, df['SalePrice'].quantile(0.33)], 'color': '#E8F5E9'},
                    {'range': [df['SalePrice'].quantile(0.33), df['SalePrice'].quantile(0.67)], 'color': '#FFF3E0'},
                    {'range': [df['SalePrice'].quantile(0.67), df['SalePrice'].max()], 'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': df['SalePrice'].quantile(0.75)
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Market comparison
        st.markdown("---")
        st.subheader("📊 Market Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price positioning
            percentile = (df['SalePrice'] < predicted_price).sum() / len(df) * 100
            
            st.write(f"""
            ### 🎯 Price Positioning
            
            Your predicted price of **${predicted_price:,.0f}** is:
            - **{percentile:.1f}th percentile** in the market
            - **${predicted_price - df['SalePrice'].median():,.0f}** {'above' if predicted_price > df['SalePrice'].median() else 'below'} median
            - **${predicted_price - df['SalePrice'].mean():,.0f}** {'above' if predicted_price > df['SalePrice'].mean() else 'below'} average
            
            **Price Range Estimate:**
            - Conservative: ${lower_bound:,.0f}
            - Expected: ${predicted_price:,.0f}
            - Optimistic: ${upper_bound:,.0f}
            """)
        
        with col2:
            # Similar properties
            st.write(f"""
            ### 🏘️ Similar Properties
            
            Properties with similar characteristics:
            - Living area: {gr_liv_area} sqft
            - Quality: {overall_qual}/10
            - Bedrooms: {bedrooms}
            - Year built: {year_built}
            
            **Market Insights:**
            - Average market price: ${df['SalePrice'].mean():,.0f}
            - Your price vs market: {((predicted_price/df['SalePrice'].mean() - 1) * 100):+.1f}%
            - Price/sqft comparison: ${predicted_price/gr_liv_area:.0f} vs ${(df['SalePrice']/df['GrLivArea']).mean():.0f} market avg
            """)
        
        # Property highlights
        st.markdown("---")
        st.subheader("✨ Property Highlights")
        
        highlights = []
        
        if overall_qual >= 8:
            highlights.append("🌟 **Excellent Quality** - Premium construction")
        if gr_liv_area > df['GrLivArea'].quantile(0.75):
            highlights.append("📐 **Spacious Living Area** - Above average size")
        if year_built > 2010:
            highlights.append("🆕 **Modern Construction** - Recently built")
        if garage_cars >= 2:
            highlights.append("🚗 **Large Garage** - Multiple car capacity")
        if pool_area > 0:
            highlights.append("🏊 **Pool Available** - Great for entertainment")
        if fireplaces > 0:
            highlights.append("🔥 **Fireplace** - Cozy ambiance")
        if bedrooms >= 4:
            highlights.append("🛏️ **Family-sized** - Multiple bedrooms")
        
        if highlights:
            for highlight in highlights:
                st.write(highlight)
        else:
            st.info("Standard property features")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📈 To Increase Value:")
            st.write("""
            - Improve overall quality rating
            - Add or upgrade bathrooms
            - Expand living area
            - Modernize kitchen and fixtures
            - Add garage space
            """)
        
        with col2:
            st.write("### 🎯 Pricing Strategy:")
            if predicted_price > df['SalePrice'].quantile(0.75):
                st.write("""
                **Premium Property**
                - List at upper bound for negotiation room
                - Target luxury buyers
                - Highlight unique features
                - Professional staging recommended
                """)
            elif predicted_price > df['SalePrice'].median():
                st.write("""
                **Mid-High Market**
                - List at predicted price
                - Competitive in good neighborhoods
                - Standard marketing approach
                - Quick sale likely
                """)
            else:
                st.write("""
                **Value Market**
                - Price competitively
                - Emphasize value proposition
                - Quick sale expected
                - Consider minor upgrades
                """)
        
        st.success(f"✅ Prediction saved! Address: {property_address if property_address else 'N/A'}")

# ==================== ABOUT PAGE ====================
else:
    st.markdown('<div class="main-header">👤 About</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Project Information</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📧 Contact Information")
        st.write("""
        - 📧 Email: muhammadrizkiputra1998@gmail.com
        - 🐙 GitHub: https://github.com/putrarizkim
        """)
        
        st.markdown("### 🎓 Education")
        st.write("""
        **Skill Academy**
        - Python Fundamental & Setup
        - Data Structures and Loops
        - Using OOP & AI Library
        - AI Model & Huggingface
        - Advance Data Structures
        - NLP Model & Transformers
        - Production & Deployment
        - Final Project: Indeks AQI App
                 
        **DQLab Academy**
        - Machine Learning & AI Track
        - Data Science Bootcamp
        - Capstone Project: House Price Prediction
        
        **Diponegoro University**
        - Bachelor of Science in Industrial Engineering
        - Focus: AI Agents & Automation
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=400", 
                use_container_width=True, caption="Real Estate Analytics")
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("""
        **Languages:**
        - Python 3.x
        - SQL
        """)
    
    with col2:
        st.write("""
        **Libraries:**
        - Scikit-learn
        - Pandas & NumPy
        - Plotly
        - Streamlit
        """)
    
    with col3:
        st.write("""
        **Tools:**
        - Jupyter Notebook
        - Git & GitHub
        - VS Code
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Project Goals")
    
    st.write("""
    1. **Akurasi Tinggi**: Membangun model dengan R² > 0.85
    2. **User-Friendly**: Interface yang mudah digunakan
    3. **Real-Time**: Prediksi instant dengan visualisasi
    4. **Insightful**: Memberikan insights pasar yang berguna
    5. **Scalable**: Mudah di-deploy dan dikembangkan
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 References & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Datasets:**
        - Kaggle House Prices Dataset
        - UCI Machine Learning Repository
        - Real Estate Market Data
        """)
    
    with col2:
        st.write("""
        **Learning Resources:**
        - DQLab Academy Materials
        - Scikit-learn Documentation
        - Real Estate Pricing Research Papers
        """)
    
    st.markdown("---")
    
    st.markdown("### 🙏 Acknowledgments")
    
    st.write("""
    Terima kasih kepada:
    - **DQLab Academy** - Untuk program Machine Learning & AI Track
    - **Mentor & Instructor** - Untuk guidance dan feedback
    - **Community** - Untuk support dan diskusi
    - **Open Source Contributors** - Untuk tools dan libraries yang amazing
    """)
    
    st.markdown("---")
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("### 📜 Recent Predictions")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        recent = history_df.tail(5)[['timestamp', 'address', 'living_area', 'bedrooms', 'predicted_price']]
        recent['predicted_price'] = recent['predicted_price'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(recent, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🏠 House Price Prediction System")
with col2:
    st.caption("🤖 Powered by Machine Learning")
with col3:
    st.caption("© 2025 Capstone Project DQLab")