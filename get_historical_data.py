import pandas as pd
import numpy as np
from meteostat import Point, Daily
from datetime import datetime, timedelta
import os
import time

class ChennaiHistoricalData:
    def __init__(self):
        self.chennai = Point(13.0827, 80.2707)
        self.data = None
        
    def fetch_historical_data(self, start_year=2000, end_year=2024):
        """Fetch historical weather data for Chennai from Meteostat with robust error handling"""
        print(f"Fetching Chennai weather data from {start_year} to {end_year}...")
        
        all_data = []
        failed_years = []
        successful_years = []
        
        for year in range(start_year, end_year + 1):
            print(f"Fetching data for {year}...")
            
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            try:
                data = Daily(self.chennai, start_date, end_date)
                data = data.fetch()
                
                if not data.empty:
                    if 'tavg' in data.columns and 'prcp' in data.columns:
                        valid_temp_days = data['tavg'].notna().sum()
                        valid_rain_days = data['prcp'].notna().sum()
                        
                        if valid_temp_days > 100 and valid_rain_days > 50:
                            all_data.append(data)
                            successful_years.append(year)
                            temp_avg = data['tavg'].mean() if data['tavg'].notna().any() else 0
                            prcp_avg = data['prcp'].mean() if data['prcp'].notna().any() else 0
                            print(f"  ✓ {year}: {len(data)} days (tavg: {temp_avg:.1f}°C, prcp: {prcp_avg:.1f}mm)")
                        else:
                            print(f"  ⚠ {year}: Poor quality data - {valid_temp_days} temp days, {valid_rain_days} rain days")
                            failed_years.append(year)
                    else:
                        print(f"  ✗ {year}: Missing required columns")
                        failed_years.append(year)
                else:
                    print(f"  ✗ {year}: No data returned")
                    failed_years.append(year)
                    
            except Exception as e:
                print(f"  ✗ {year}: Error - {str(e)[:100]}...")
                failed_years.append(year)
            
            time.sleep(2)
        
        if all_data:
            self.data = pd.concat(all_data)
            print(f"\n✅ Successfully fetched {len(self.data)} days from {len(successful_years)} years: {successful_years}")
            print(f"❌ Failed years: {failed_years}")
            
            return self.data
        else:
            print("❌ No usable data could be fetched")
            return None
    
    def clean_data(self):
        """Improved data cleaning with better handling of missing values"""
        if self.data is None or self.data.empty:
            print("No data to clean")
            return None
        
        print("\nCleaning and validating data...")
        
        df = self.data.copy()
        
        # Reset index and rename time column
        if df.index.name == 'time':
            df = df.reset_index()
            df.rename(columns={'time': 'date'}, inplace=True)
        
        print(f"Initial data shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Show initial missing values
        print("\nInitial missing values:")
        for col in df.columns:
            if col != 'date':
                missing = df[col].isna().sum()
                if missing > 0:
                    print(f"  {col}: {missing} missing values ({missing/len(df):.1%})")
        
        # Clean temperature columns
        temp_columns = ['tavg', 'tmin', 'tmax']
        for col in temp_columns:
            if col in df.columns:
                # Remove extreme outliers for Chennai
                df[col] = df[col].apply(lambda x: x if pd.notna(x) and 15 <= x <= 45 else np.nan)
        
        # Clean precipitation
        if 'prcp' in df.columns:
            df['prcp'] = df['prcp'].apply(lambda x: max(0, x) if pd.notna(x) else np.nan)
            df['prcp'] = df['prcp'].apply(lambda x: min(x, 300) if pd.notna(x) else np.nan)
        
        # Clean pressure
        if 'pres' in df.columns:
            df['pres'] = df['pres'].apply(lambda x: x if pd.notna(x) and 995 <= x <= 1025 else np.nan)
        
        # Clean wind speed
        if 'wspd' in df.columns:
            df['wspd'] = df['wspd'].apply(lambda x: x if pd.notna(x) and 0 <= x <= 50 else np.nan)
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Improved imputation
        df = self.impute_missing_values(df)
        
        self.data = df
        return df
    
    def impute_missing_values(self, df):
        """Fixed imputation without deprecated methods"""
        print("\nImputing missing values...")
        
        # Show missing values before imputation
        print("Missing values before imputation:")
        for col in df.select_dtypes(include=[np.number]).columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing")
        
        # Strategy 1: Seasonal averages for temperatures
        for col in ['tavg', 'tmin', 'tmax']:
            if col in df.columns:
                # Use monthly averages to fill missing temperatures
                monthly_avg = df.groupby('month')[col].transform('mean')
                df[col] = df[col].fillna(monthly_avg)
        
        # Strategy 2: For precipitation, use 0 for dry season, monthly avg for wet season
        if 'prcp' in df.columns:
            dry_season = df['month'].isin([1, 2, 3, 4, 5, 12])
            df.loc[dry_season, 'prcp'] = df.loc[dry_season, 'prcp'].fillna(0)
            
            # For remaining missing, use monthly average
            monthly_prcp = df.groupby('month')['prcp'].transform('mean')
            df['prcp'] = df['prcp'].fillna(monthly_prcp)
        
        # Strategy 3: For other columns, use forward fill then backward fill (new method)
        other_cols = ['pres', 'wspd', 'tsun']
        for col in other_cols:
            if col in df.columns:
                # Use new ffill()/bfill() methods instead of deprecated fillna(method=...)
                df[col] = df[col].ffill().bfill()
        
        # Final pass: for any remaining NaNs, use column median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Show results
        remaining_missing = df.isnull().sum().sum()
        print(f"Remaining missing values after imputation: {remaining_missing}")
        
        if remaining_missing > 0:
            print("Columns with remaining missing values:")
            for col in df.columns:
                missing = df[col].isna().sum()
                if missing > 0:
                    print(f"  {col}: {missing} missing")
        
        return df
    
    def validate_data(self):
        """Simple data validation"""
        if self.data is None:
            return
        
        df = self.data
        print("\nData Validation:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Years: {sorted(df['year'].unique())}")
        
        print("\nColumn statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['year', 'month', 'day', 'day_of_year']:
                continue
            if col in df.columns:
                print(f"  {col}: mean={df[col].mean():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    def save_cleaned_data(self, filename='chennai_weather_cleaned.csv'):
        """Save cleaned data to CSV"""
        output_dir = 'historical_data'
        os.makedirs(output_dir, exist_ok=True)
        
        if self.data is None:
            print("No data to save")
            return None
        
        filepath = os.path.join(output_dir, filename)
        self.data.to_csv(filepath, index=False)
        print(f"✅ Cleaned data saved to: {filepath}")
        print(f"   Records: {len(self.data)}, Columns: {len(self.data.columns)}")
        
        return filepath

def test_meteostat_availability():
    """Test function to check what years are available in Meteostat"""
    print("Testing Meteostat availability for Chennai...")
    print("="*50)
    
    chennai = Point(13.0827, 80.2707)
    test_years = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2015, 2010, 2005, 2000,1999,1998,1997,1995,1993,1990]
    
    available_years = []
    
    for year in test_years:
        try:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            data = Daily(chennai, start_date, end_date)
            data = data.fetch()
            
            if not data.empty and 'tavg' in data.columns and 'prcp' in data.columns:
                valid_days = data['tavg'].notna().sum()
                if valid_days > 100:
                    available_years.append(year)
                    print(f"✓ {year}: Available ({valid_days} days)")
                else:
                    print(f"⚠ {year}: Limited data ({valid_days} days)")
            else:
                print(f"✗ {year}: No data")
                
        except Exception as e:
            print(f"✗ {year}: Error - {str(e)[:50]}")
        
        time.sleep(1)
    
    print(f"\n📊 Available years: {available_years}")
    return available_years

def main():
    """Main function"""
    print("Chennai Historical Weather Data Pipeline")
    print("="*50)
    
    # Test availability first
    print("\nStep 0: Testing Meteostat availability...")
    available_years = test_meteostat_availability()
    
    if not available_years:
        print("❌ No years available from Meteostat.")
        return
    
    # Initialize processor
    processor = ChennaiHistoricalData()
    
    # Use available years range
    start_year = min(available_years)
    end_year = max(available_years)
    
    print(f"\nStep 1: Fetching data from {start_year} to {end_year}...")
    raw_data = processor.fetch_historical_data(start_year, end_year)
    
    if raw_data is None:
        print("❌ Failed to fetch data.")
        return
    
    # Clean data
    print("\nStep 2: Cleaning data...")
    cleaned_data = processor.clean_data()
    
    if cleaned_data is None:
        print("❌ Data cleaning failed.")
        return
    
    # Validate
    print("\nStep 3: Validating data...")
    processor.validate_data()
    
    # Save
    print("\nStep 4: Saving data...")
    saved_path = processor.save_cleaned_data()
    
    print(f"\n🎉 Success! Data pipeline completed.")
    print(f"📁 Cleaned dataset: {saved_path}")

if __name__ == "__main__":
    main()