import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Delhi Real Estate AI",
    page_icon="🏠",
    layout="wide"
)

# --- LOADER FUNCTION ---
@st.cache_resource
def load_data():
    # Find the model file relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'delhi_house_data.pkl')
    
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

# --- INITIALIZE APP ---
data = load_data()

if not data:
    st.error("⚠️ CRITICAL ERROR: Model file not found!")
    st.warning("Please run 'python build_data.py' first to generate the necessary data.")
    st.stop()

# Unpack data
model = data['model']
metro_df = data['metro_data']
school_df = data['school_data']
hospital_df = data['hospital_data']
aqi_df = data['aqi_data']
location_map = data.get('location_map', {})

# --- INTELLIGENCE ENGINE ---
def get_nearest_feature(user_lat, user_lon, df_target, name_col, value_col=None):
    if df_target.empty:
        return 0.0, "N/A", 0
    
    # 1. Setup Coordinates
    user_coords = np.radians([[user_lat, user_lon]])
    target_coords = np.radians(df_target[['Latitude', 'Longitude']])
    
    # 2. Find Nearest Neighbor
    tree = BallTree(target_coords, metric='haversine')
    dist, idx = tree.query(user_coords, k=1)
    nearest_index = idx[0][0]
    
    # 3. Extract Info
    distance_km = dist[0][0] * 6371 # Earth radius
    name = df_target.iloc[nearest_index][name_col]
    
    value = 0
    if value_col:
        value = df_target.iloc[nearest_index][value_col]
        
    return distance_km, name, value

# --- USER INTERFACE ---
st.title("🏠 Delhi NCR Smart Property Predictor")
st.markdown("### 🤖 AI-Powered Market Valuation & Livability Analysis")
st.divider()

# Layout Columns
col_input, col_results = st.columns([1, 1.5], gap="medium")

with col_input:
    st.header("1. Property Details")
    area = st.number_input("📏 Area (Sq. Ft.)", min_value=100, max_value=50000, value=1250, step=50)
    
    c1, c2 = st.columns(2)
    bedrooms = c1.slider("🛏️ Bedrooms", 1, 10, 3)
    bathrooms = c2.slider("🚿 Bathrooms", 1, 10, 2)
    
    st.write("")
    st.header("2. Location")
    
    # Tabs for location selection
    tab_list, tab_manual = st.tabs(["📍 Select Area", "🗺️ Manual Coords"])
    
    # Defaults (Connaught Place)
    sel_lat, sel_lon = 28.6304, 77.2177
    
    with tab_list:
        if location_map:
            # Filter short/empty names and sort
            valid_locs = sorted([k for k in location_map.keys() if len(str(k)) > 3])
            loc_name = st.selectbox("Choose Locality:", valid_locs)
            
            # Update coords based on selection
            if loc_name:
                coords = location_map[loc_name]
                sel_lat = coords['latitude']
                sel_lon = coords['longitude']
                st.caption(f"Coordinates: {sel_lat:.4f}, {sel_lon:.4f}")
        else:
            st.error("Location map is empty. Re-run build_data.py")

    with tab_manual:
        man_lat = st.number_input("Latitude", value=28.6304, format="%.5f")
        man_lon = st.number_input("Longitude", value=77.2177, format="%.5f")

    # Logic: Use manual if changed from default, otherwise use selected
    if man_lat != 28.6304 or man_lon != 77.2177:
        final_lat = man_lat
        final_lon = man_lon
    else:
        final_lat = sel_lat
        final_lon = sel_lon

with col_results:
    st.header("3. Market Analysis")
    
    if st.button("🚀 Analyze & Predict Price", type="primary", use_container_width=True):
        
        with st.spinner("🔍 Analyzing Satellite Data, Metro Lines, and Amenities..."):
            # A. Calculate Intelligence
            m_dist, m_name, _ = get_nearest_feature(final_lat, final_lon, metro_df, 'Station')
            s_dist, s_name, _ = get_nearest_feature(final_lat, final_lon, school_df, 'Name')
            h_dist, h_name, _ = get_nearest_feature(final_lat, final_lon, hospital_df, 'Name')
            _, a_name, local_aqi = get_nearest_feature(final_lat, final_lon, aqi_df, 'Name', 'AQI')
            
            # B. Prepare Input Vector (Must match training order)
            input_df = pd.DataFrame([[
                area, bedrooms, bathrooms, m_dist, local_aqi, s_dist, h_dist
            ]], columns=['area', 'Bedrooms', 'Bathrooms', 'Metro_Dist_km', 'Local_AQI', 'Nearest_School_Dist', 'Nearest_Hospital_Dist'])
            
            # C. Predict
            price = model.predict(input_df)[0]
            
            # --- DISPLAY RESULTS ---
            st.success("✅ Analysis Complete")
            
            # 1. Price Card
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #ddd; text-align: center;">
                <p style="margin:0; color: #555; font-size: 1.1em;">Estimated Market Value</p>
                <h1 style="margin:0; color: #2e7d32; font-size: 3.5em;">₹ {price/100000:.2f} Lakhs</h1>
                <p style="margin:0; color: #888;">(₹ {price:,.0f})</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.subheader("🏙️ Livability Report")
            
            # 2. Metro Section
            with st.expander(f"🚇 Nearest Metro: **{m_name}**", expanded=True):
                col_m1, col_m2 = st.columns([1, 3])
                col_m1.metric("Distance", f"{m_dist:.2f} km")
                if m_dist < 1.0:
                    col_m2.success("**Premium Connectivity!** Walking distance adds significant value.")
                elif m_dist < 3.0:
                    col_m2.info("**Good Connectivity.** Short commute to station.")
                else:
                    col_m2.warning("**Low Connectivity.** Reliance on private transport.")

            # 3. Amenities Grid
            c_school, c_hosp, c_aqi = st.columns(3)
            
            with c_school:
                st.metric("🏫 School", f"{s_dist:.2f} km")
                st.caption(f"{s_name}")
                
            with c_hosp:
                st.metric("🏥 Hospital", f"{h_dist:.2f} km")
                st.caption(f"{h_name}")
                
            with c_aqi:
                st.metric("🌫️ AQI", f"{local_aqi:.0f}")
                if local_aqi > 300:
                    st.error("Hazardous")
                elif local_aqi > 200:
                    st.warning("Poor")
                else:
                    st.success("Moderate")