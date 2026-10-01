import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(
    page_title="Custom House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dynamic House Price Prediction & Analytics")
st.markdown("Upload a dataset or use default database records to train models and estimate property prices.")

# File Uploader Section
st.sidebar.header("📁 Dataset Source")
uploaded_file = st.sidebar.file_uploader("Upload custom Housing CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        try:
            return pd.read_csv("Housing.csv")
        except Exception:
            conn = sqlite3.connect("housing_database.db")
            df = pd.read_sql("SELECT * FROM housing", conn)
            conn.close()
            return df

df = load_data(uploaded_file)

if uploaded_file is not None:
    st.sidebar.success("Custom CSV file loaded successfully!")
else:
    st.sidebar.info("Using default local database / Housing.csv.")

# Preprocessing & Model Training
feature_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Ensure required columns exist
missing_cols = [col for col in feature_cols + ['price'] if col not in df.columns]

if missing_cols:
    st.error(f"Uploaded file is missing required columns: {missing_cols}")
else:
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Input Form
    st.subheader("📝 Input Property Details for Valuation")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Property Area (sq ft)", min_value=100, max_value=25000, value=int(df['area'].median()), step=100)
        bedrooms = st.slider("Bedrooms", int(df['bedrooms'].min()), int(df['bedrooms'].max()), int(df['bedrooms'].median()))

    with col2:
        bathrooms = st.slider("Bathrooms", int(df['bathrooms'].min()), int(df['bathrooms'].max()), int(df['bathrooms'].median()))
        stories = st.slider("Stories / Floors", int(df['stories'].min()), int(df['stories'].max()), int(df['stories'].median()))

    with col3:
        parking = st.selectbox("Parking Spaces", sorted(df['parking'].unique().tolist()))
        predict_btn = st.button("🔮 Calculate Valuation", use_container_width=True)

    st.markdown("---")

    if predict_btn or 'pred_done' not in st.session_state:
        st.session_state['pred_done'] = True
        
        input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        res_a, res_b = st.columns([1, 2])
        with res_a:
            st.metric(label="Predicted Price", value=f"₹ {prediction:,.2f}")
        with res_b:
            st.success(f"Valuation calculated based on **{len(df)} records** in current dataset.")

    st.markdown("---")
    with st.expander("📊 View Current Active Dataset"):
        st.dataframe(df.head(15), use_container_width=True)