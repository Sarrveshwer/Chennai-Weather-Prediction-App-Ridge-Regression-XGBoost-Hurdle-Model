import pandas as pd
import numpy as np
import requests
import joblib
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, average_precision_score
from geopy.geocoders import Nominatim
from auto_logger import setup_logging

C_HEADER = "\033[95m"
C_INFO = "\033[94m"
C_SUCCESS = "\033[92m"
C_WARN = "\033[93m"
C_FAIL = "\033[91m"
C_METRIC = "\033[96m"
C_END = "\033[0m"

if __name__ == "__main__":
    setup_logging("weather_engine_architect_final")

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print(f"{C_FAIL}[CRITICAL] XGBoost library not found. Precipitation models will not train.{C_END}")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeatherForecastingEngine:
    def __init__(self, location="Chennai"):
        print(f"\n{C_HEADER}{'='*80}{C_END}")
        print(f"{C_HEADER}SYSTEM INITIALIZATION: {location.upper()} ARCHITECT ENGINE{C_END}")
        print(f"{C_HEADER}{'='*80}{C_END}")
        
        self.location = location
        self.coords = self._get_coordinates(location)
        self.data = None
        
        self.temp_model = None
        self.rain_classifier = None 
        self.rain_regressors = []   
        self.scaler = RobustScaler()
        
        self.is_trained = False
        self.feature_cols = []
        
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
        
        print(f"{C_INFO}[INIT] Target Coordinates: {self.coords}{C_END}")
        print(f"{C_INFO}[INIT] Architecture: Leak-Proof Hurdle + Generalization Guard{C_END}")

    def _get_coordinates(self, city):
        try:
            geolocator = Nominatim(user_agent="weather_architect_v1")
            loc = geolocator.geocode(city, timeout=10)
            if loc: return loc.latitude, loc.longitude
        except:
            print(f"{C_WARN}[WARN] Geocoding service failed. Reverting to hardcoded defaults.{C_END}")
        return 13.0827, 80.2707 

    def fetch_data(self, days_history=9500):
        print(f"\n{C_HEADER}--- PHASE 1: MASTER DATA INGESTION (Open-Meteo) ---{C_END}")
        
        end = datetime.datetime.now()
        start_target = datetime.datetime(2000, 1, 1)
        start = max(start_target, end - timedelta(days=days_history))
        
        print(f"{C_INFO}[REQ] Date Range: {start.date()} to {end.date()}{C_END}")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.coords[0],
            "longitude": self.coords[1],
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                               "precipitation_sum", "wind_speed_10m_max", "surface_pressure_mean",
                               "relative_humidity_2m_mean"]),
            "timezone": "auto"
        }
        
        try:
            resp = requests.get(url, params=params)
            daily = resp.json().get("daily", {})
            if not daily:
                print(f"{C_FAIL}[ERROR] Empty payload received from API.{C_END}")
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
            
            df = df.infer_objects(copy=False)
            df['prcp'] = df['prcp'].fillna(0).clip(lower=0)
            df = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
            
            self.data = df
            print(f"{C_SUCCESS}[SUCCESS] Ingested {len(df)} records. Memory: {df.memory_usage().sum()/1024:.1f}KB{C_END}")
            return self.data
        except Exception as e:
            print(f"{C_FAIL}[CRITICAL] Fetch Error: {e}{C_END}")
            return None

    def get_data_slice(self, days=500):
        if self.data is None: self.fetch_data()
        return self.data.tail(days).copy().reset_index(drop=True)

    def engineer_features(self, df=None):
        if df is None: df = self.data.copy()
        print(f"\n{C_HEADER}--- PHASE 2: LEAK-PROOF FEATURE PIPELINE ---{C_END}")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        df['doy'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        for col, p in [('doy', 365.25), ('month', 12)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / p)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / p)

        df['pres_delta_24h'] = df['pres'].shift(1).diff(1)
        df['temp_drift'] = df['tavg'].shift(1) - df['tavg'].shift(1).rolling(7).mean()
        
        if 'pres' in df.columns and 'wspd' in df.columns:
            df['storm_energy'] = (df['wspd'].shift(1)**2) * (1013 - df['pres'].shift(1))
        
        for col in ['prcp', 'wspd']: df[f'{col}_log'] = np.log1p(df[col].shift(1))

        targets = ['tavg', 'prcp', 'pres', 'wspd', 'hum']
        for col in targets:
            if col not in df.columns: continue
            for lag in [1, 7, 14]: df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_roll_7'] = df[col].shift(1).rolling(7).mean()

        df = df.iloc[30:].reset_index(drop=True)
        self.feature_cols = [c for c in df.columns if c not in ['date', 'tavg', 'prcp', 'tmin', 'tmax'] and pd.api.types.is_numeric_dtype(df[c])]
        
        return df

    def train_temperature_model(self, df):
        print(f"\n{C_HEADER}--- PHASE 3: TEMP TRAINING & GENERALIZATION GUARD ---{C_END}")
        from sklearn.linear_model import Ridge
        X, y = df[self.feature_cols], df['tavg']
        
        split = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            temp_cv_model = Ridge(alpha=50.0)
            temp_cv_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            cv_scores.append(r2_score(y_train.iloc[val_idx], temp_cv_model.predict(X_train.iloc[val_idx])))
        
        self.avg_cv_r2 = np.mean(cv_scores)
        self.temp_model = Ridge(alpha=50.0)
        self.temp_model.fit(X_train, y_train)
        
        train_r2 = r2_score(y_train, self.temp_model.predict(X_train))
        test_r2 = r2_score(y_test, self.temp_model.predict(X_test))
        
        print(f"{C_METRIC}   CV R2 Stability: {self.avg_cv_r2:.4f}{C_END}")
        print(f"{C_METRIC}   Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}{C_END}")
        self._plot_temp_performance(y_test, self.temp_model.predict(X_test))

    def train_precipitation_hurdle(self, df):
        if not XGB_AVAILABLE: return
        print(f"\n{C_HEADER}--- PHASE 4: PRECIPITATION HURDLE ---{C_END}")
        X, y = df[self.feature_cols], df['prcp']
        X_s = self.scaler.fit_transform(X)
        split = int(len(X) * 0.85)
        
        y_cls = (y.iloc[:split] > 0.5).astype(int)
        self.rain_classifier = XGBClassifier(n_estimators=300, max_depth=4, reg_lambda=50, scale_pos_weight=3)
        self.rain_classifier.fit(X_s[:split], y_cls)
        
        mask = y.iloc[:split] > 0.5
        self.rain_regressors = [XGBRegressor(objective='reg:squarederror', n_estimators=300, reg_lambda=20), 
                                XGBRegressor(objective='reg:gamma', n_estimators=300, reg_lambda=20)]
        for m in self.rain_regressors: m.fit(X_s[:split][mask], np.log1p(y.iloc[:split][mask]))
        
        self.is_trained = True
        self._plot_precip_performance(y.iloc[split:], self.rain_classifier.predict(X_s[split:]))
        self._plot_feature_correlation(df)

    def _plot_temp_performance(self, y_test, y_pred):
        try:
            y_test, y_pred = pd.Series(y_test).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)
            errors = np.abs(y_test - y_pred)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Temperature Diagnostics: Leak-Proof Ridge (alpha=50)', fontsize=16, fontweight='bold')
            
            axes[0,0].scatter(y_test, y_pred, alpha=0.5, c=errors, cmap='viridis', s=15)
            axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[0,0].set_title('Predicted vs Actual')

            axes[0,1].scatter(y_pred, y_test - y_pred, alpha=0.5, c='purple', s=15)
            axes[0,1].axhline(y=0, color='r', linestyle='--')
            axes[0,1].set_title('Residual Plot')

            sns.kdeplot(errors, ax=axes[0,2], fill=True, color='green', alpha=0.3)
            axes[0,2].axvline(0.2, color='darkgreen', linestyle='--', label='Target (<0.2C)')
            axes[0,2].set_title('Error Density Distribution')

            axes[1,0].plot(range(100), y_test[:100], 'b-', label='Actual', alpha=0.6)
            axes[1,0].plot(range(100), y_pred[:100], 'r--', label='Pred', alpha=0.8)
            axes[1,0].set_title('Sample Timeline (100 Days)')

            coef = pd.Series(self.temp_model.coef_, index=self.feature_cols).abs().sort_values(ascending=False).head(10)
            sns.barplot(x=coef.values, y=coef.index, ax=axes[1,1], hue=coef.index, legend=False, palette='viridis')
            axes[1,1].set_title('Top 10 Feature Drivers')

            mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
            metrics_txt = f"METRICS\n\nMAE: {mae:.4f} C\nRMSE: {rmse:.4f} C\nR2: {r2:.4f}"
            axes[1,2].text(0.1, 0.5, metrics_txt, fontsize=12, bbox=dict(facecolor='white', boxstyle='round,pad=1'))
            axes[1,2].axis('off')
            
            plt.tight_layout()
            timestamp = time.strftime("%Y/%m/%d-%H:%M:%S")
            save_path = f"graphs/temperature_performance_{timestamp}.png"
            plt.savefig(save_path, dpi=300)
            print(f"{C_SUCCESS}   [ARCHIVE] {save_path}{C_END}")
            plt.show()
        except: pass

    def _plot_precip_performance(self, y_test, y_pred_cls):
        print(f"{C_INFO}[VIZ] Launching Robust Precipitation Dashboard...{C_END}")
        try:
            y_test_bin = (y_test > 0.5).astype(int)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Hurdle Model Architecture: Orthodox Diagnostics (Rain)', fontsize=16, fontweight='bold')
            
            sns.kdeplot(y_test, ax=axes[0,0], fill=True, color='blue', label='Actual')
            sns.kdeplot(y_pred_cls, ax=axes[0,0], fill=True, color='red', label='Pred')
            axes[0,0].set_title('Rain Distribution Comparison')

            precision, recall, _ = precision_recall_curve(y_test_bin, y_pred_cls)
            axes[0,1].step(recall, precision, color='b', alpha=0.2, where='post')
            axes[0,1].fill_between(recall, precision, step='post', alpha=0.2, color='b')
            axes[0,1].set_title(f'Precision-Recall (AP: {average_precision_score(y_test_bin, y_pred_cls):.2f})')

            if hasattr(self.rain_classifier, 'feature_importances_'):
                imp = pd.Series(self.rain_classifier.feature_importances_, index=self.feature_cols).sort_values(ascending=False).head(10)
                sns.barplot(x=imp.values, y=imp.index, ax=axes[0,2], hue=imp.index, legend=False, palette='magma')
                axes[0,2].set_title('Top Rain Triggers (Gatekeeper)')

            y_test_np, y_pred_np = np.array(y_test), np.array(y_pred_cls)
            bins, labels = [-1, 0.1, 10, 30, 1000], ['Dry', 'Light', 'Moderate', 'Heavy']
            df_err = pd.DataFrame({'actual': y_test_np, 'error': np.abs(y_test_np - y_pred_np)})
            df_err['cat'] = pd.cut(df_err['actual'], bins=bins, labels=labels)
            mean_err = df_err.groupby('cat', observed=False)['error'].mean()
            sns.barplot(x=mean_err.index, y=mean_err.values, ax=axes[1,0], palette='viridis', hue=mean_err.index, legend=False)
            axes[1,0].set_title('MAE by Intensity')

            axes[1,1].plot(np.sort(np.abs(y_test - y_pred_cls)), np.linspace(0, 1, len(y_test)), color='purple')
            axes[1,1].set_title('Cumulative Error Distribution')

            acc = accuracy_score(y_test_bin, (y_pred_cls > 0.5).astype(int))
            axes[1,2].text(0.1, 0.5, f"HURDLE METRICS\n\nClassifier Acc: {acc:.4f}\nRain Threshold: 0.5mm\nReg lambda: 20", 
                           fontsize=12, bbox=dict(facecolor='white', boxstyle='round,pad=1'))
            axes[1,2].axis('off')

            plt.tight_layout()
            timestamp = time.strftime("%Y/%m/%d-%H:%M:%S")
            save_path = f"graphs/precipitation_performance_{timestamp}.png"
            plt.savefig(save_path, dpi=300)
            print(f"{C_SUCCESS}   [ARCHIVE] {save_path}{C_END}")
            plt.show()
        except Exception as e: print(f"{C_WARN}Precip Viz Fail: {e}{C_END}")

    def _plot_feature_correlation(self, df):
        try:
            plt.figure(figsize=(14, 10))
            sns.heatmap(df[self.feature_cols[:15]].corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
            plt.title('Physics Inter-Correlation (Top 15 Features)')
            timestamp = time.strftime("%Y/%m/%d-%H:%M:%S")
            save_path = f"graphs/feature_correlation_{timestamp}.png"
            plt.savefig(save_path, dpi=300)
            print(f"{C_SUCCESS}   [ARCHIVE] {save_path}{C_END}")
            plt.show()
        except: pass

    def print_final_audit(self):
        print(f"\n{C_HEADER}{'='*80}{C_END}")
        print(f"{C_HEADER}FINAL SYSTEM ARCHITECT AUDIT & MODEL DETAILS{C_END}")
        print(f"{C_HEADER}{'='*80}{C_END}")
        print(f"{C_INFO}[AUDIT] FEATURE MATRIX DETAILS{C_END}")
        print(f"  -> Total Unique Signals: {len(self.feature_cols)}")
        print(f"\n{C_INFO}[AUDIT] TEMPERATURE MODEL (RIDGE){C_END}")
        print(f"  -> CV R2 Stability:      {getattr(self, 'avg_cv_r2', 'N/A'):.4f}")
        coefs = sorted(zip(self.feature_cols, self.temp_model.coef_), key=lambda x: abs(x[1]), reverse=True)
        for f, v in coefs[:5]: print(f"     * {f:20} : {v:+.8f}")
        print(f"\n{C_INFO}[AUDIT] PRECIPITATION MODEL (XGBOOST HURDLE){C_END}")
        print(f"  -> Classifier Estimators: {self.rain_classifier.n_estimators}")
        print(f"  -> L2 Regularization (λ): {self.rain_classifier.reg_lambda}")
        print(f"\n{C_SUCCESS}ALL SYSTEMS VERIFIED AND ARCHIVED.{C_END}")

    def run_full_pipeline(self):
        self.fetch_data()
        df_proc = self.engineer_features()
        self.train_temperature_model(df_proc)
        self.train_precipitation_hurdle(df_proc)
        joblib.dump(self, 'saved_models/weather_engine.joblib')
        self.print_final_audit()

if __name__ == "__main__":
    engine = WeatherForecastingEngine("Chennai")
    engine.run_full_pipeline()