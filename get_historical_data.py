import pandas as pd
import numpy as np
import requests
import os
from auto_logger import setup_logging

# Initialize Logger
if __name__ == "__main__":
    setup_logging("historical_data_pipeline")

class ChennaiHistoricalData:
    def __init__(self):
        self.lat = 13.0827
        self.lon = 80.2707
        self.data = None
        
    def fetch_historical_data(self, start_year=2000, end_year=2024):
        print(f"--- Fetching Data {start_year}-{end_year} (Open-Meteo) ---")
        
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                   "precipitation_sum", "wind_speed_10m_max", "surface_pressure_mean",
                   "relative_humidity_2m_mean"]),
            "timezone": "auto"
        }
        
        try:
            print(f"Requesting {start_date} to {end_date}...")
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"[ERROR] API Status: {response.status_code}")
                return None
                
            json_data = response.json()
            daily = json_data.get("daily", {})
            
            if not daily:
                print("[ERROR] No data returned.")
                return None
            
            df = pd.DataFrame({
                'date': daily.get("time"),
                'tavg': daily.get("temperature_2m_mean"),
                'tmax': daily.get("temperature_2m_max"),
                'tmin': daily.get("temperature_2m_min"),
                'prcp': daily.get("precipitation_sum"),
                'wspd': daily.get("wind_speed_10m_max"),
                'pres': daily.get("surface_pressure_mean"),
                'hum': daily.get("relative_humidity_2m_mean") 
            })
            
            df['date'] = pd.to_datetime(df['date'])
            print(f"Fetched {len(df)} records.")
            self.data = df
            return self.data
            
        except Exception as e:
            print(f"[CRITICAL] {e}")
            return None
    
    def clean_data(self):
        if self.data is None or self.data.empty:
            print("No data to clean")
            return None
        
        print("\n--- Cleaning Data ---")
        df = self.data.copy()
        
        for col in ['tavg', 'tmax', 'tmin']:
            df[col] = df[col].apply(lambda x: x if 15 <= x <= 50 else np.nan)
            
        df['prcp'] = df['prcp'].clip(lower=0)
        df['hum'] = df['hum'].clip(0, 100)
        
        df = df.interpolate(method='linear', limit_direction='both')
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        self.data = df
        return df

    def save_cleaned_data(self, filename='chennai_weather_cleaned.csv'):
        output_dir = 'historical_data'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        self.data.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath

def main():
    processor = ChennaiHistoricalData()
    raw_data = processor.fetch_historical_data(2000, 2024)
    
    if raw_data is not None:
        processor.clean_data()
        processor.save_cleaned_data()
        print("\n[SUCCESS] Pipeline Complete.")

if __name__ == "__main__":
    main()