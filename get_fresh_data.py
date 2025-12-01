from meteostat import Point, Daily
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="weather_app_v2")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            print(f"{C_GREEN}Found location: {location.address}{C_RESET}")
            return location.latitude, location.longitude, location.address
        else:
            print(f"{C_RED}City not found.{C_RESET}")
            return None, None, None
    except GeocoderTimedOut:
        print(f"{C_RED}Geocoding service timed out.{C_RESET}")
        return None, None, None

def get_meteostat_data(city_name="Chennai", days=600):
    lat, lon, full_name = get_coordinates(city_name)
    if lat is None:
        print(f"{C_YELLOW}Could not resolve {city_name}. Defaulting to Chennai.{C_RESET}")
        lat, lon = 13.0827, 80.2707
        
    location_point = Point(lat, lon)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"{C_BLUE}Fetching data for {city_name} ({lat:.4f}, {lon:.4f}) from {start_date.date()} to {end_date.date()}{C_RESET}")
    
    data = Daily(location_point, start_date, end_date)
    data = data.fetch()
    
    if data.empty:
        print(f"{C_RED}No data received from meteostat{C_RESET}")
        return [], "Unknown", "Unknown"
    
    last_fetched_date = data.index[-1].date()
    today_date = end_date.date()
    is_today_present = (last_fetched_date == today_date)
    
    if is_today_present:
        print(f"{C_GREEN}Meteostat returned data for today ({today_date}){C_RESET}")
    else:
        print(f"{C_YELLOW}Meteostat data ends on {last_fetched_date}. Today's data will be imputed.{C_RESET}")
    
    if 'tavg' in data.columns:
        data['tavg'] = data['tavg'].fillna(data['tavg'].mean())
    if 'prcp' in data.columns:
        data['prcp'] = data['prcp'].fillna(0)
    if 'pres' in data.columns:
        data['pres'] = data['pres'].fillna(1013.0)
    if 'wspd' in data.columns:
        data['wspd'] = data['wspd'].fillna(data['wspd'].mean())

    weather_data = []
    
    for date, row in data.iterrows():
        temp = row['tavg'] if pd.notna(row['tavg']) else 15.0
        
        is_rainy = row['prcp'] > 0.5
        
        if is_rainy:
            base_hum = 85.0
        elif temp > 30: 
            base_hum = 40.0 
        elif temp < 10:
            base_hum = 70.0 
        else:
            base_hum = 60.0
            
        final_humidity = max(30, min(99, base_hum + np.random.randint(-10, 10)))

        weather_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'tavg': temp,
            'tmin': row['tmin'] if pd.notna(row['tmin']) else temp - 5,
            'tmax': row['tmax'] if pd.notna(row['tmax']) else temp + 5,
            'prcp': row['prcp'] if pd.notna(row['prcp']) else 0.0,
            'pres': row['pres'] if pd.notna(row['pres']) else 1013.0,
            'wspd': row['wspd'] if pd.notna(row['wspd']) else 10.0,
            'hum': final_humidity,
            'year': date.year,
            'month': date.month,
            'day': date.day
        })
    
    return weather_data, full_name, "India" if "India" in str(full_name) else "Global"

def create_ml_features(df):
    print(f"{C_BLUE}Creating comprehensive ML features...{C_RESET}")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    df['day_of_year_rad'] = 2 * np.pi * df['day_of_year'] / 365.25
    for h in [1, 2, 3]:
        df[f'day_sin_{h}'] = np.sin(h * df['day_of_year_rad'])
        df[f'day_cos_{h}'] = np.cos(h * df['day_of_year_rad'])
    
    for col in ['tavg', 'prcp', 'pres', 'wspd']:
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    for col in ['tavg', 'prcp']:
        for w in [7, 30]:
            df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean().shift(1)

    df['recent_rain_volume'] = df['prcp'].rolling(30).sum().shift(1)
    df['is_rainy_season'] = (df['recent_rain_volume'] > df['recent_rain_volume'].median()).astype(int)
    
    df['is_monsoon'] = df['is_rainy_season'] 
    df['monsoon_strength'] = df['is_rainy_season'] * 0.8
    
    df['is_winter'] = (df['tavg'] < df['tavg'].rolling(365).mean()).astype(int)
    df['winter_strength'] = df['is_winter'] * 1.0

    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def json_to_csv(city="Chennai", output_file='recent_weather.csv'):
    print(f"{C_BLUE}Getting weather data for {city}...{C_RESET}")
    weather_data, full_name, country = get_meteostat_data(city, 730)
    
    if not weather_data:
        print(f"{C_RED}No data received{C_RESET}")
        return None, None
    
    df = pd.DataFrame(weather_data)
    df = create_ml_features(df)
    
    df.to_csv(output_file, index=False)
    print(f"{C_GREEN}Saved data for {full_name} to {output_file}{C_RESET}")
    
    return df, country

if __name__ == "__main__":
    json_to_csv("Chennai")