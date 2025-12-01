import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

class UltimateMAEPredictor:
    def __init__(self):
        self.models = []
        self.scaler = RobustScaler()
        self.is_trained = False
        self.feature_columns = None
        
    def load_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path, parse_dates=["date"])
        print(f"Loaded {len(self.data)} daily records")
        return self.data

    def clean_dataset(self, df):
        print("Applying Physics-Based Data Cleaning...")
        
        # 1. Constraint Enforcement (Physics)
        if 'prcp' in df.columns:
            df['prcp'] = df['prcp'].clip(lower=0)  # No negative rain
        
        if 'wspd' in df.columns:
            df['wspd'] = df['wspd'].clip(lower=0)  # No negative wind
        
        if 'pres' in df.columns:
            # Standard sea level pressure range
            df.loc[(df['pres'] < 870) | (df['pres'] > 1090), 'pres'] = np.nan
            
        if 'tavg' in df.columns:
            # Reasonable temp range for Earth surface
            df.loc[(df['tavg'] < -60) | (df['tavg'] > 60), 'tavg'] = np.nan

        # 2. Smart Imputation (Time-Series Interpolation)
        # Fills gaps based on the trend of surrounding days
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # Final safety fill for edge cases
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df

    def feature_engineering_pipeline(self):
        print("Generating advanced meteorological features...")
        df = self.data.copy()
        df = self.clean_dataset(df)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Log-transform skewed features for better model digestion
        for col in ['wspd', 'pres']:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
        
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        cols_to_lag = ['tavg', 'pres', 'wspd', 'prcp']
        cols_to_lag = [c for c in cols_to_lag if c in df.columns]

        for col in cols_to_lag:
            for lag in [1, 2, 3, 7, 14, 30]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            for w in [3, 7, 14, 30]:
                df[f'{col}_mean_{w}'] = df[col].shift(1).rolling(w).mean()
                df[f'{col}_std_{w}'] = df[col].shift(1).rolling(w).std()
                
                if col == 'prcp':
                    df[f'{col}_max_{w}'] = df[col].shift(1).rolling(w).max()

        if 'tavg' in df.columns and 'pres' in df.columns:
            df['temp_pres_interaction'] = df['tavg'] * np.log1p(df['pres'])
        
        if 'wspd' in df.columns and 'pres' in df.columns:
            df['storm_energy'] = (df['wspd'] ** 2) * (1013 - df['pres'])

        if 'prcp' in df.columns:
            rain_mask = df['prcp'].shift(1) > 0.1
            # Groupby cumsum to calculate streaks correctly
            df['rain_streak'] = rain_mask.groupby((rain_mask != rain_mask.shift()).cumsum()).cumsum()
            
            df['prcp_skew_30'] = df['prcp'].shift(1).rolling(30).skew()
            df['prcp_kurt_30'] = df['prcp'].shift(1).rolling(30).kurt()

        df = df.dropna(subset=['prcp'])
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        df = df.iloc[30:].reset_index(drop=True)
        
        return df

    def prepare_features(self, target_column='prcp'):
        df = self.feature_engineering_pipeline()
        
        exclude = ['date', target_column, 'snow', 'wpgt', 'tsun']
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        X = df[features].copy()
        y = df[target_column].copy()
        
        print(f"  Input Features: {len(features)}")
        print(f"  Training Samples: {len(X)}")
        
        self.feature_columns = features
        return X, y

    def get_ensemble_configs(self, seed):
        configs = []
        
        # 1. Tweedie (Targeting Zero-Inflation)
        configs.append(XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=1.6, 
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.8,
            min_child_weight=3,
            n_jobs=-1,
            random_state=seed + 1,
            tree_method='hist'
        ))

        # 2. Squared Error (Targeting RMSE/Cyclones)
        configs.append(XGBRegressor(
            objective='reg:squarederror',
            n_estimators=2500,
            learning_rate=0.015,
            max_depth=10,
            subsample=0.6,
            colsample_bytree=0.7,
            gamma=0.1,
            n_jobs=-1,
            random_state=seed + 2,
            tree_method='hist'
        ))
        
        # 3. Poisson (Count Data Specialist)
        configs.append(XGBRegressor(
            objective='count:poisson',
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed + 3,
            tree_method='hist'
        ))

        return configs

    def create_performance_plots(self, y_test, y_pred):
        print("Generating diagnostics...")
        try:
            if hasattr(y_test, 'reset_index'): y_test = y_test.reset_index(drop=True)
            y_pred = pd.Series(y_pred)
            residuals = y_test - y_pred
            abs_errors = np.abs(residuals)

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Ensemble Model Performance (Log-Transformed Training)', fontsize=18, fontweight='bold')

            # 1. Scatter
            sc = axes[0, 0].scatter(y_test, y_pred, alpha=0.6, c=abs_errors, cmap='coolwarm', s=30)
            axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], 'k--', lw=2)
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].set_xlabel('Actual (mm)')
            axes[0, 0].set_ylabel('Predicted (mm)')
            plt.colorbar(sc, ax=axes[0, 0], label='Error')

            # 2. Residuals
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5, color='teal', s=20)
            axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_title('Residuals')
            axes[0, 1].set_ylabel('Residual (mm)')

            # 3. Dist
            sns.histplot(abs_errors, bins=50, kde=True, color='purple', ax=axes[0, 2])
            axes[0, 2].set_title('Error Distribution')

            # 4. Timeline
            limit = 150
            axes[1, 0].plot(range(limit), y_test[:limit], label='Actual', color='black', alpha=0.7)
            axes[1, 0].plot(range(limit), y_pred[:limit], label='Predicted', color='blue', linestyle='--', alpha=0.8)
            axes[1, 0].legend()
            axes[1, 0].set_title('Forecast Sample')

            # 5. Binned Error
            bins = [-1, 0.1, 10, 30, 1000]
            labels = ['Dry', 'Light', 'Moderate', 'Heavy']
            df_err = pd.DataFrame({'actual': y_test, 'error': abs_errors})
            df_err['cat'] = pd.cut(df_err['actual'], bins=bins, labels=labels)
            mean_err = df_err.groupby('cat')['error'].mean()
            sns.barplot(x=mean_err.index, y=mean_err.values, ax=axes[1, 1], palette='viridis')
            axes[1, 1].set_title('Error by Intensity')

            # 6. Stats
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            txt = f"MAE: {mae:.4f} mm\nRMSE: {rmse:.4f} mm\nR2: {r2:.4f}"
            axes[1, 2].text(0.5, 0.5, txt, fontsize=18, ha='center')
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig('precipitation_performance.png', dpi=300)
            print("✅ Saved precipitation_performance.png")
            plt.close(fig)
        except Exception as e:
            print(f"Plot error: {e}")

    def train_ultimate_ensemble(self, test_size=0.1):
        if not XGB_AVAILABLE: return None
        
        X, y = self.prepare_features(target_column='prcp')
        
        # --- DATA ENGINEERING: LOG TRANSFORM ---
        # Train on Log(y+1) to reduce the impact of massive outliers (Cyclones)
        # This stabilizes gradient descent and lowers global RMSE
        y_log = np.log1p(y)
        
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train_log, y_test_log = y_log.iloc[:split], y_log.iloc[split:]
        y_test_actual = y.iloc[split:]
        
        print(f"Train Size: {len(X_train)} | Test Size: {len(X_test)}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.models = []
        ensemble_preds = []
        
        configs = self.get_ensemble_configs(seed=42)
        
        print(f"Training {len(configs)} specialized XGBoost models...")
        
        for i, model in enumerate(configs):
            print(f"Training Model {i+1} ({model.objective})...")
            
            # Tweedie and Poisson handle 0s naturally, but log-space is also good.
            # We use log-transformed targets for stability.
            
            model.fit(
                X_train_scaled, y_train_log,
                eval_set=[(X_test_scaled, y_test_log)],
                verbose=False
            )
            
            # Predict in log space, then inverse transform
            pred_log = model.predict(X_test_scaled)
            pred = np.expm1(pred_log) 
            pred = np.maximum(pred, 0)
            
            ensemble_preds.append(pred)
            self.models.append(model)
            
            print(f"  -> Model {i+1} MAE: {mean_absolute_error(y_test_actual, pred):.4f}")

        # Weighted Average
        y_pred = (0.4 * ensemble_preds[0]) + (0.3 * ensemble_preds[1]) + (0.3 * ensemble_preds[2])
        
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        
        self.is_trained = True
        self.feature_columns = X.columns.tolist()
        
        self.create_performance_plots(y_test_actual, y_pred)
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def evaluate_mae_performance(self, metrics):
        print("\n" + "="*50)
        print(f"FINAL ENSEMBLE METRICS:")
        print(f"MAE:  {metrics['mae']:.4f} mm")
        print(f"RMSE: {metrics['rmse']:.4f} mm")
        print(f"R2:   {metrics['r2']:.4f}")
        print("="*50)
        
        if self.models:
            print("\nTOP FEATURES (Model 1 - Tweedie):")
            imp = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': self.models[0].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            print(imp.to_string(index=False))

    def save_ultimate_model(self):
        if not self.is_trained: return
        os.makedirs('saved_models', exist_ok=True)
        path = 'saved_models/precipitator.joblib'
        # Save with a flag indicating log-transformation was used
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_ensemble': True,
            'target_transform': 'log1p' 
        }, path)
        print(f"Ensemble saved to {path}")

def main():
    print("Initializing XGBoost Ensemble with Data Cleaning...")
    predictor = UltimateMAEPredictor()
    path = 'historical_data/chennai_weather_cleaned.csv'
    if not os.path.exists(path): path = 'recent_weather.csv'
    
    try:
        predictor.load_data(path)
        metrics = predictor.train_ultimate_ensemble()
        if metrics:
            predictor.evaluate_mae_performance(metrics)
            predictor.save_ultimate_model()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()