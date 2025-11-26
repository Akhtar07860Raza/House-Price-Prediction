import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import requests
import pickle
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestRegressor
import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'delhi_house_data.pkl')

print(f"🚀 STARTING ROBUST SYSTEM BUILD...")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 1. LOAD METRO DATA (Offline) ---
print("\n[1/5] Loading Metro Data...")
metro_path = os.path.join(DATA_DIR, 'DELHI_METRO_DATA.csv')
if os.path.exists(metro_path):
    df_metro = pd.read_csv(metro_path)
    df_metro = df_metro.dropna(subset=['Latitude', 'Longitude'])
    print(f"   ✅ Loaded {len(df_metro)} Metro Stations")
else:
    print(f"   ❌ ERROR: File not found at {metro_path}")
    exit()

# --- 2. LOAD AQI DATA (Offline) ---
print("\n[2/5] Parsing AQI Data...")
aqi_path = os.path.join(DATA_DIR, 'data_aqi_cpcb.xml')
if os.path.exists(aqi_path):
    try:
        tree = ET.parse(aqi_path)
        root = tree.getroot()
        aqi_list = []
        for station in root.iter('Station'):
            try:
                aqi_list.append({
                    'Name': station.get('id'),
                    'Latitude': float(station.get('latitude')),
                    'Longitude': float(station.get('longitude')),
                    'AQI': float(station.find('Air_Quality_Index').get('Value'))
                })
            except: continue
        df_aqi = pd.DataFrame(aqi_list).dropna(subset=['Latitude', 'Longitude'])
        print(f"   ✅ Loaded {len(df_aqi)} AQI Stations")
    except:
        df_aqi = pd.DataFrame(columns=['Name', 'Latitude', 'Longitude', 'AQI'])
else:
    df_aqi = pd.DataFrame(columns=['Name', 'Latitude', 'Longitude', 'AQI'])

# --- 3. FETCH SCHOOLS & HOSPITALS (Online + Backup) ---
print("\n[3/5] Fetching Schools & Hospitals...")

# BACKUP DATA (Used if internet fails)
backup_amenities = [
    {'Name': 'AIIMS Hospital', 'Type': 'hospital', 'Latitude': 28.5659, 'Longitude': 77.2111},
    {'Name': 'Safdarjung Hospital', 'Type': 'hospital', 'Latitude': 28.5680, 'Longitude': 77.2058},
    {'Name': 'Max Super Speciality', 'Type': 'hospital', 'Latitude': 28.6290, 'Longitude': 77.2786},
    {'Name': 'Fortis Escorts', 'Type': 'hospital', 'Latitude': 28.5603, 'Longitude': 77.2722},
    {'Name': 'Sir Ganga Ram Hospital', 'Type': 'hospital', 'Latitude': 28.6383, 'Longitude': 77.1882},
    {'Name': 'BLK Super Speciality', 'Type': 'hospital', 'Latitude': 28.6437, 'Longitude': 77.1796},
    {'Name': 'Apollo Hospital', 'Type': 'hospital', 'Latitude': 28.5395, 'Longitude': 77.2862},
    {'Name': 'DPS RK Puram', 'Type': 'school', 'Latitude': 28.5733, 'Longitude': 77.1767},
    {'Name': 'Modern School', 'Type': 'school', 'Latitude': 28.6289, 'Longitude': 77.2282},
    {'Name': 'Springdales School', 'Type': 'school', 'Latitude': 28.6432, 'Longitude': 77.1821},
    {'Name': 'Sanskriti School', 'Type': 'school', 'Latitude': 28.5933, 'Longitude': 77.1869},
    {'Name': 'The Mother\'s International', 'Type': 'school', 'Latitude': 28.5406, 'Longitude': 77.2042},
    {'Name': 'Vasant Valley School', 'Type': 'school', 'Latitude': 28.5359, 'Longitude': 77.1427},
    {'Name': 'Bal Bhaarti Public School', 'Type': 'school', 'Latitude': 28.6397, 'Longitude': 77.1839},
    {'Name': 'St. Columba\'s School', 'Type': 'school', 'Latitude': 28.6293, 'Longitude': 77.2096}
]

overpass_url = "http://overpass-api.de/api/interpreter"
overpass_query = """[out:json];(node["amenity"="school"](28.2,76.8,28.9,77.8);node["amenity"="hospital"](28.2,76.8,28.9,77.8););out center;"""

try:
    # Try fetching from internet (timeout after 10 seconds)
    response = requests.get(overpass_url, params={'data': overpass_query}, timeout=15)
    data = response.json()
    amenity_list = []
    for el in data['elements']:
        if 'tags' in el and 'name' in el['tags']:
            amenity_list.append({'Name': el['tags']['name'], 'Type': el['tags']['amenity'], 'Latitude': el['lat'], 'Longitude': el['lon']})
    
    # If we got results, use them. If not, use backup.
    if len(amenity_list) > 0:
        df_amenities = pd.DataFrame(amenity_list)
        print(f"   ✅ Internet Fetch Success: {len(amenity_list)} locations found.")
    else:
        raise Exception("Empty response")

except Exception as e:
    print(f"   ⚠️ Internet fetch failed/timed out. USING BACKUP DATA. ({e})")
    df_amenities = pd.DataFrame(backup_amenities)

# Process Dataframes
df_schools = df_amenities[df_amenities['Type'] == 'school'].reset_index(drop=True)
df_hospitals = df_amenities[df_amenities['Type'] == 'hospital'].reset_index(drop=True)
print(f"   📊 Final Count: {len(df_schools)} Schools, {len(df_hospitals)} Hospitals")

# --- 4. TRAIN MODEL ---
print("\n[4/5] Training Model...")
housing_path = os.path.join(DATA_DIR, 'Delhi_v2.csv')
if os.path.exists(housing_path):
    df_housing = pd.read_csv(housing_path).dropna(subset=['latitude', 'longitude', 'price', 'area'])
    df_housing = df_housing[(df_housing['price'] > 500000) & (df_housing['price'] < 500000000)]
    print(f"   ✅ Loaded {len(df_housing)} Housing Records")
else:
    print(f"   ❌ ERROR: File not found at {housing_path}")
    exit()

print("   🗺️ Building Location Map...")
location_map = df_housing.groupby('Address')[['latitude', 'longitude']].mean().to_dict('index')

def get_nearest(df_main, df_target, val_col=None):
    if df_target.empty: return np.zeros(len(df_main))
    rad_main = np.radians(df_main[['latitude', 'longitude']])
    rad_target = np.radians(df_target[['Latitude', 'Longitude']])
    tree = BallTree(rad_target, metric='haversine')
    dist, idx = tree.query(rad_main, k=1)
    if val_col: return df_target.iloc[idx.flatten()][val_col].values
    return dist.flatten() * 6371

print("   Calculating distances...")
df_housing['Metro_Dist_km'] = get_nearest(df_housing, df_metro)
df_housing['Nearest_School_Dist'] = get_nearest(df_housing, df_schools)
df_housing['Nearest_Hospital_Dist'] = get_nearest(df_housing, df_hospitals)
df_housing['Local_AQI'] = get_nearest(df_housing, df_aqi, 'AQI') if not df_aqi.empty else 150

X = df_housing[['area', 'Bedrooms', 'Bathrooms', 'Metro_Dist_km', 'Local_AQI', 'Nearest_School_Dist', 'Nearest_Hospital_Dist']].fillna(0)
y = df_housing['price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("   ✅ Model Trained!")

# --- 5. SAVE ---
print(f"\n[5/5] Saving to {MODEL_FILE}...")
try:
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'metro_data': df_metro, 'school_data': df_schools, 'hospital_data': df_hospitals, 'aqi_data': df_aqi, 'location_map': location_map}, f)
    print(f"🎉 SUCCESS! File saved.")
except Exception as e:
    print(f"❌ ERROR SAVING: {e}")