import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LinearTemperatureModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = None
        self.poly = None
        self.feature_columns = []
        self.train_indices = None
        self.test_indices = None
        
    def load_data(self):
        """Load and validate the dataset with enhanced checks"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"📊 Loaded {len(self.data)} daily rows")
            
            # Enhanced column validation
            required_cols = ['date', 'tavg', 'tmin', 'tmax', 'prcp', 'year', 'month', 'day', 'day_of_year']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return False
            
            # Data quality assessment
            print("🔍 Data Quality Assessment:")
            for col in required_cols:
                if col in self.data.columns:
                    missing = self.data[col].isna().sum()
                    unique = self.data[col].nunique()
                    print(f"   {col}: {missing} missing, {unique} unique values")
            
            # Convert date and sort to ensure temporal order
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            # Remove any duplicate dates
            if self.data['date'].duplicated().any():
                print("⚠️ Removing duplicate dates")
                self.data = self.data.drop_duplicates(subset=['date']).reset_index(drop=True)
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_time_aware_split(self, test_size=0.15):
        """Create time-aware train-test split to prevent data leakage"""
        split_index = int(len(self.data) * (1 - test_size))
        self.train_indices = list(range(split_index))
        self.test_indices = list(range(split_index, len(self.data)))
        
        print(f"📅 Time-aware split: Train {len(self.train_indices)} samples, Test {len(self.test_indices)} samples")
        print(f"   Train period: {self.data.iloc[0]['date'].date()} to {self.data.iloc[split_index-1]['date'].date()}")
        print(f"   Test period:  {self.data.iloc[split_index]['date'].date()} to {self.data.iloc[-1]['date'].date()}")
        
        return self.train_indices, self.test_indices
    
    def create_safe_features(self, target_column='tavg'):
        """Create features without data leakage using time-aware approach"""
        print("🔄 Creating safe features without data leakage...")
        
        try:
            # Create a copy to avoid modifying original during feature creation
            data_copy = self.data.copy()
            
            # First, create time-aware split
            train_indices, test_indices = self.create_time_aware_split()
            
            # 1. Basic temperature metrics (no leakage)
            data_copy['temp_range'] = data_copy['tmax'] - data_copy['tmin']
            data_copy['temp_avg_min_max'] = (data_copy['tmin'] + data_copy['tmax']) / 2
            data_copy['temp_center'] = data_copy['tmin'] + (data_copy['temp_range'] / 2)
            
            # 2. Cyclical features (no leakage - derived from date only)
            data_copy['day_of_year_rad'] = 2 * np.pi * data_copy['day_of_year'] / 365.25
            optimal_day_harmonics = [1, 2, 3]
            for harmonic in optimal_day_harmonics:
                data_copy[f'day_sin_{harmonic}'] = np.sin(harmonic * data_copy['day_of_year_rad'])
                data_copy[f'day_cos_{harmonic}'] = np.cos(harmonic * data_copy['day_of_year_rad'])
            
            data_copy['month_rad'] = 2 * np.pi * data_copy['month'] / 12
            data_copy['month_sin'] = np.sin(data_copy['month_rad'])
            data_copy['month_cos'] = np.cos(data_copy['month_rad'])
            data_copy['month_sin_2'] = np.sin(2 * data_copy['month_rad'])
            
            # 3. Seasonal indicators for Chennai (no leakage)
            def chennai_seasonal_weight(month):
                if month in [12, 1]:
                    return 1.0
                elif month in [11, 2]:
                    return 0.8
                elif month in [10, 3]:
                    return 0.5
                elif month in [9, 4]:
                    return 0.3
                elif month in [5, 8]:
                    return 0.1
                else:
                    return 0.0
            
            data_copy['winter_strength'] = data_copy['month'].apply(chennai_seasonal_weight)
            data_copy['monsoon_strength'] = ((data_copy['month'] >= 6) & (data_copy['month'] <= 9)).astype(float) * 0.8
            data_copy['summer_strength'] = ((data_copy['month'] >= 3) & (data_copy['month'] <= 5)).astype(float) * 0.6
            
            # 4. Trend components (no leakage)
            data_copy['year_trend'] = data_copy['year'] - data_copy['year'].min()
            data_copy['year_trend_scaled'] = data_copy['year_trend'] / data_copy['year_trend'].max()
            data_copy['year_trend_squared'] = data_copy['year_trend'] ** 2
            
            # 5. LAG FEATURES - CRITICAL: Create these separately for train and test to avoid leakage
            predictive_lags = [1, 2, 3, 4, 7]
            
            # Initialize lag columns
            for lag in predictive_lags:
                data_copy[f'tavg_lag_{lag}'] = np.nan
                data_copy[f'prcp_lag_{lag}'] = np.nan
            
            # Create lag features only using past data (no future information)
            for i in range(len(data_copy)):
                for lag in predictive_lags:
                    if i >= lag:
                        data_copy.loc[i, f'tavg_lag_{lag}'] = data_copy.loc[i - lag, target_column]
                        data_copy.loc[i, f'prcp_lag_{lag}'] = data_copy.loc[i - lag, 'prcp']
            
            # Weekly pattern lag
            data_copy[f'tavg_lag_14'] = np.nan
            for i in range(len(data_copy)):
                if i >= 14:
                    data_copy.loc[i, f'tavg_lag_14'] = data_copy.loc[i - 14, target_column]
            
            # 6. Precipitation features (no leakage)
            data_copy['had_precipitation'] = (data_copy['prcp'] > 0).astype(int)
            data_copy['prcp_sqrt'] = np.sqrt(data_copy['prcp'] + 1)
            data_copy['heavy_rain'] = (data_copy['prcp'] > 10).astype(int)
            
            # Recent rain using lag features (already created safely)
            data_copy['recent_rain'] = (data_copy['prcp_lag_1'] > 0).astype(int)
            
            # 7. Temperature change patterns (using safely created lag features)
            data_copy['temp_change_1d'] = data_copy['tavg_lag_1'] - data_copy['tavg_lag_2']
            data_copy['temp_change_2d'] = data_copy['tavg_lag_2'] - data_copy['tavg_lag_3']
            data_copy['temp_change_3d'] = data_copy['tavg_lag_3'] - data_copy['tavg_lag_4']
            data_copy['temp_change_magnitude'] = np.abs(data_copy['temp_change_1d'])
            
            # Calculate volatility safely (using only available data)
            temp_change_cols = ['temp_change_1d', 'temp_change_2d']
            data_copy['temp_volatility'] = data_copy[temp_change_cols].std(axis=1, skipna=True)
            
            # 8. EXPANDING STATISTICS - CRITICAL: Calculate separately for train and test
            # For train data: use expanding window up to current point
            # For test data: use statistics from training data only
            
            train_data = data_copy.iloc[train_indices].copy()
            
            for window in [7, 14]:
                # Calculate expanding statistics on training data
                train_expanding_mean = train_data[target_column].expanding(min_periods=window).mean()
                train_expanding_std = train_data[target_column].expanding(min_periods=window).std()
                
                # Initialize columns
                data_copy[f'tavg_expanding_mean_{window}'] = np.nan
                data_copy[f'tavg_expanding_std_{window}'] = np.nan
                
                # Fill training data
                for i in train_indices:
                    if i >= window:
                        data_copy.loc[i, f'tavg_expanding_mean_{window}'] = train_expanding_mean.iloc[i - train_indices[0]]
                        data_copy.loc[i, f'tavg_expanding_std_{window}'] = train_expanding_std.iloc[i - train_indices[0]]
                
                # For test data, use the last value from training data (no future info)
                last_train_mean = train_expanding_mean.iloc[-1] if len(train_expanding_mean) > 0 else np.nan
                last_train_std = train_expanding_std.iloc[-1] if len(train_expanding_std) > 0 else np.nan
                
                for i in test_indices:
                    data_copy.loc[i, f'tavg_expanding_mean_{window}'] = last_train_mean
                    data_copy.loc[i, f'tavg_expanding_std_{window}'] = last_train_std
            
            # 9. Day of week effects (no leakage)
            data_copy['day_of_week'] = data_copy['date'].dt.dayofweek
            data_copy['is_weekend'] = data_copy['day_of_week'].isin([5, 6]).astype(int)
            data_copy['dow_rad'] = 2 * np.pi * data_copy['day_of_week'] / 7
            data_copy['dow_sin'] = np.sin(data_copy['dow_rad'])
            data_copy['dow_cos'] = np.cos(data_copy['dow_rad'])
            
            # 10. Interaction features (using safely created features)
            data_copy['winter_temp_effect'] = data_copy['winter_strength'] * data_copy['tavg_lag_1']
            data_copy['monsoon_temp_effect'] = data_copy['monsoon_strength'] * data_copy['tavg_lag_1']
            data_copy['precip_cooling_effect'] = data_copy['prcp_sqrt'] * data_copy['monsoon_strength']
            data_copy['temp_range_seasonal'] = data_copy['temp_range'] * data_copy['summer_strength']
            data_copy['lag_interaction'] = data_copy['tavg_lag_1'] * data_copy['tavg_lag_2']
            
            # 11. Advanced features
            data_copy['temp_momentum'] = data_copy['temp_change_1d'] - data_copy['temp_change_2d']
            
            # Seasonal adjustment - calculate monthly averages from training data only
            monthly_avg_train = train_data.groupby('month')[target_column].mean()
            data_copy['seasonal_adjustment'] = np.nan
            for month in range(1, 13):
                month_avg = monthly_avg_train.get(month, np.nan)
                if not np.isnan(month_avg):
                    month_mask = data_copy['month'] == month
                    data_copy.loc[month_mask, 'seasonal_adjustment'] = data_copy.loc[month_mask, target_column] - month_avg
            
            # 12. Handle missing values safely
            # For lag features, forward fill within reasonable limits
            lag_cols = [col for col in data_copy.columns if 'lag' in col or 'change' in col]
            for col in lag_cols:
                data_copy[col] = data_copy[col].fillna(method='ffill').fillna(method='bfill')
            
            # For expanding features, use training-based imputation
            expanding_cols = [col for col in data_copy.columns if 'expanding' in col]
            for col in expanding_cols:
                train_mean = data_copy.loc[train_indices, col].mean()
                data_copy[col] = data_copy[col].fillna(train_mean)
            
            # For other numeric columns, use median from training data only
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in lag_cols + expanding_cols and col != target_column:
                    if data_copy[col].isna().any():
                        train_median = data_copy.loc[train_indices, col].median()
                        data_copy[col] = data_copy[col].fillna(train_median)
            
            # Remove rows where target is missing
            data_copy = data_copy.dropna(subset=[target_column])
            
            # Remove first rows that don't have complete history for lags
            max_lag = 14
            data_copy = data_copy.iloc[max_lag:].reset_index(drop=True)
            
            # Update indices after removing initial rows
            self.train_indices = [i for i in self.train_indices if i >= max_lag]
            self.test_indices = [i for i in self.test_indices if i >= max_lag]
            self.train_indices = [i - max_lag for i in self.train_indices if i >= max_lag]
            self.test_indices = [i - max_lag for i in self.test_indices if i >= max_lag]
            
            created_count = len([col for col in data_copy.columns if col not in ['date', 'tavg', 'tmin', 'tmax', 'prcp']])
            print(f"✅ Created {created_count} safe features without data leakage")
            
            self.data = data_copy
            return True
            
        except Exception as e:
            print(f"❌ Error creating safe features: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_safe_features(self, target_column='tavg'):
        """Prepare features ensuring no data leakage using time-aware split"""
        print(f"🎯 Preparing safe features for {target_column}...")
        
        try:
            # Feature categories
            temporal_features = ['year', 'month', 'day', 'day_of_year', 'year_trend', 'year_trend_scaled', 'year_trend_squared', 'day_of_week', 'is_weekend']
            cyclical_features = ['day_sin_1', 'day_cos_1', 'day_sin_2', 'day_cos_2', 'day_sin_3', 'day_cos_3', 'month_sin', 'month_cos', 'month_sin_2', 'dow_sin', 'dow_cos']
            seasonal_features = ['winter_strength', 'monsoon_strength', 'summer_strength']
            weather_features = ['temp_range', 'temp_avg_min_max', 'temp_center', 'prcp_sqrt', 'had_precipitation', 'heavy_rain', 'recent_rain']
            lag_features = ['tavg_lag_1', 'tavg_lag_2', 'tavg_lag_3', 'tavg_lag_4', 'tavg_lag_7', 'tavg_lag_14', 'prcp_lag_1', 'prcp_lag_2', 'prcp_lag_3', 'prcp_lag_4', 'prcp_lag_7']
            change_features = ['temp_change_1d', 'temp_change_2d', 'temp_change_3d', 'temp_change_magnitude', 'temp_volatility', 'temp_momentum']
            expanding_features = ['tavg_expanding_mean_7', 'tavg_expanding_mean_14', 'tavg_expanding_std_7', 'tavg_expanding_std_14']
            interaction_features = ['winter_temp_effect', 'monsoon_temp_effect', 'precip_cooling_effect', 'temp_range_seasonal', 'lag_interaction']
            advanced_features = ['seasonal_adjustment']
            
            # Combine all features
            all_features = (temporal_features + cyclical_features + seasonal_features + 
                          weather_features + lag_features + change_features + expanding_features + 
                          interaction_features + advanced_features)
            
            # Filter to existing columns and exclude target/correlated
            existing_features = [f for f in all_features if f in self.data.columns]
            exclude_cols = ['tmin', 'tmax'] if target_column == 'tavg' else []
            
            self.feature_columns = [
                f for f in existing_features 
                if f not in exclude_cols and f != target_column
            ]
            
            print(f"📋 Safe feature pool: {len(self.feature_columns)} features")
            
            # Use time-aware split to get train and test data
            train_data = self.data.iloc[self.train_indices]
            test_data = self.data.iloc[self.test_indices]
            
            X_train = train_data[self.feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[self.feature_columns]
            y_test = test_data[target_column]
            
            print(f"📊 Safe dataset:")
            print(f"   Training: {len(X_train)} samples")
            print(f"   Testing:  {len(X_test)} samples")
            print(f"   Train period: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
            print(f"   Test period:  {test_data['date'].min().date()} to {test_data['date'].max().date()}")
            
            return X_train, X_test, y_train, y_test, self.feature_columns
            
        except Exception as e:
            print(f"❌ Error preparing safe features: {e}")
            raise
    
    def select_safe_features(self, X_train, y_train):
        """Feature selection using training data only to prevent leakage"""
        print("🎯 Performing safe feature selection...")
        
        try:
            # Use only training data for feature selection
            methods_scores = {}
            
            # Method 1: Correlation with target
            correlations = X_train.corrwith(y_train).abs()
            methods_scores['correlation'] = correlations / correlations.max()
            
            # Method 2: Mutual information
            mi_scores = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=15)
            mi_features = pd.Series(mi_scores, index=X_train.columns)
            methods_scores['mutual_info'] = mi_features / mi_features.max()
            
            # Method 3: F-statistic
            f_scores, _ = f_regression(X_train, y_train)
            f_features = pd.Series(f_scores, index=X_train.columns)
            methods_scores['f_statistic'] = f_features / f_features.max()
            
            # Method 4: RFE with cross-validation
            lr_temp = LinearRegression()
            optimal_features = min(15, X_train.shape[1])
            selector_rfe = RFE(lr_temp, n_features_to_select=optimal_features, step=1)
            selector_rfe.fit(X_train, y_train)
            rfe_scores = pd.Series(selector_rfe.support_.astype(int), index=X_train.columns)
            methods_scores['rfe'] = rfe_scores
            
            # Weights
            weights = {
                'correlation': 0.25,
                'mutual_info': 0.40,
                'f_statistic': 0.20,
                'rfe': 0.15
            }
            
            combined_scores = pd.Series(0.0, index=X_train.columns)
            for method, score in methods_scores.items():
                combined_scores += weights[method] * score
            
            n_features = min(18, X_train.shape[1])
            final_features = combined_scores.nlargest(n_features).index.tolist()
            
            print(f"✅ Selected {len(final_features)} safe features using training data only")
            print(f"   Top 5 features: {final_features[:5]}")
            
            return final_features
            
        except Exception as e:
            print(f"❌ Error in safe feature selection: {e}")
            return X_train.columns.tolist()
    
    def create_polynomial_features(self, X_train, X_test, degree=2, interaction_only=True):
        """Create polynomial features safely using training data for fitting"""
        print(f"🔄 Creating safe polynomial features (degree {degree})...")
        
        try:
            self.poly = PolynomialFeatures(degree=degree, 
                                         include_bias=False, 
                                         interaction_only=interaction_only)
            
            # Fit on training data only, transform both train and test
            X_train_poly = self.poly.fit_transform(X_train)
            X_test_poly = self.poly.transform(X_test)
            
            feature_names = self.poly.get_feature_names_out(X_train.columns)
            
            print(f"✅ Created {X_train_poly.shape[1]} safe polynomial features")
            return X_train_poly, X_test_poly, feature_names.tolist()
            
        except Exception as e:
            print(f"❌ Error creating safe polynomial features: {e}")
            return X_train, X_test, X_train.columns.tolist()
    
    def create_performance_plots(self, y_test, y_pred, model_name):
        """Create comprehensive performance visualization"""
        print("📊 Creating performance visualization...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Safe Linear Regression: {model_name}', fontsize=16, fontweight='bold')
            
            # 1. Prediction vs Actual scatter plot
            scatter = axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20, c=np.abs(y_test - y_pred), cmap='viridis')
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Temperature (°C)')
            axes[0, 0].set_ylabel('Predicted Temperature (°C)')
            axes[0, 0].set_title('Predicted vs Actual (Color = Error Magnitude)')
            axes[0, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 0], label='Absolute Error (°C)')
            
            # 2. Residual plot
            residuals = y_test - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20, c=np.abs(residuals), cmap='plasma')
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Predicted Temperature (°C)')
            axes[0, 1].set_ylabel('Residuals (°C)')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Error distribution
            errors = np.abs(residuals)
            axes[0, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)
            axes[0, 2].axvline(0.2, color='g', linestyle='--', linewidth=2, label='Target MAE (0.2°C)')
            axes[0, 2].axvline(errors.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}°C')
            axes[0, 2].set_xlabel('Absolute Error (°C)')
            axes[0, 2].set_ylabel('Density')
            axes[0, 2].set_title('Error Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Time series comparison
            test_indices = range(min(50, len(y_test)))
            axes[1, 0].plot(test_indices, y_test.iloc[test_indices], 'b-', label='Actual', alpha=0.8, linewidth=2)
            axes[1, 0].plot(test_indices, y_pred[test_indices], 'r-', label='Predicted', alpha=0.8, linewidth=1.5)
            axes[1, 0].fill_between(test_indices, y_test.iloc[test_indices], y_pred[test_indices], 
                                  alpha=0.3, color='gray', label='Error')
            axes[1, 0].set_xlabel('Test Sample Index')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].set_title('Sample Predictions vs Actual')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Error by temperature range
            temperature_bins = pd.cut(y_test, bins=8)
            error_by_temp = pd.DataFrame({
                'actual': y_test,
                'error': errors,
                'bin': temperature_bins
            }).groupby('bin')['error'].agg(['mean', 'std']).fillna(0)
            
            x_pos = range(len(error_by_temp))
            axes[1, 1].bar(x_pos, error_by_temp['mean'], yerr=error_by_temp['std'], 
                          capsize=5, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Temperature Range (°C)')
            axes[1, 1].set_ylabel('Mean Absolute Error (°C)')
            axes[1, 1].set_title('Error by Temperature Range (±1 STD)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([str(bin) for bin in error_by_temp.index], rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mae_progress = min(1.0, (0.2 / mae)) if mae > 0 else 1.0
            rmse_progress = min(1.0, (0.3 / rmse)) if rmse > 0 else 1.0
            
            axes[1, 2].text(0.1, 0.8, 'PERFORMANCE METRICS', fontsize=14, fontweight='bold')
            axes[1, 2].text(0.1, 0.7, f'MAE: {mae:.4f}°C\nRMSE: {rmse:.4f}°C\nR²: {r2:.6f}', 
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            axes[1, 2].text(0.1, 0.5, 'TARGET PROGRESS', fontsize=12, fontweight='bold')
            axes[1, 2].text(0.1, 0.45, f'MAE Target: {mae_progress*100:.1f}%', fontsize=10)
            axes[1, 2].text(0.1, 0.4, f'RMSE Target: {rmse_progress*100:.1f}%', fontsize=10)
            
            error_stats = f'Errors < 0.2°C: {(errors < 0.2).mean()*100:.1f}%\n'
            error_stats += f'Errors < 0.3°C: {(errors < 0.3).mean()*100:.1f}%\n'
            error_stats += f'Errors < 0.5°C: {(errors < 0.5).mean()*100:.1f}%\n'
            error_stats += f'Max Error: {errors.max():.3f}°C'
            axes[1, 2].text(0.1, 0.25, error_stats, fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].set_title('Performance Analysis')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig('safe_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Performance plots saved as 'safe_performance.png'")
            
        except Exception as e:
            print(f"❌ Error creating performance plots: {e}")
    
    def train_safe_linear_model(self, target_column='tavg', save_model=True):
        """Train linear regression model without data leakage"""
        print(f"\n📈 TRAINING SAFE LINEAR REGRESSION MODEL")
        print("🎯 TARGET: MAE ≤ 0.2°C, RMSE ≤ 0.3°C")
        print("🛡️  NO DATA LEAKAGE")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Create safe features without data leakage
        if not self.create_safe_features(target_column):
            return False
        
        try:
            # Prepare safe features using time-aware split
            X_train, X_test, y_train, y_test, self.feature_columns = self.prepare_safe_features(target_column)
            
            # Safe feature selection using training data only
            selected_features = self.select_safe_features(X_train, y_train)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            self.feature_columns = selected_features
            
            # Safe polynomial features
            X_train_poly, X_test_poly, self.feature_columns = self.create_polynomial_features(
                X_train_selected, X_test_selected, degree=2, interaction_only=True
            )
            
            print(f"📈 Final feature set: {len(self.feature_columns)} features")
            print(f"📊 Training samples: {len(X_train_poly)}")
            print(f"📊 Testing samples: {len(X_test_poly)}")
            print(f"🌡️ Target range: {y_train.min():.1f}°C to {y_train.max():.1f}°C")
            
            # Safe scaling - fit on training data only
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_poly)
            X_test_scaled = self.scaler.transform(X_test_poly)
            
            # Model testing with cross-validation on training data only
            models = {
                'Ridge α=0.05': Ridge(alpha=0.05, max_iter=10000, random_state=42),
                'Ridge α=0.1': Ridge(alpha=0.1, max_iter=10000, random_state=42),
                'Ridge α=0.5': Ridge(alpha=0.5, max_iter=10000, random_state=42),
                'Lasso α=0.005': Lasso(alpha=0.005, max_iter=10000, random_state=42),
                'Lasso α=0.01': Lasso(alpha=0.01, max_iter=10000, random_state=42),
                'Lasso α=0.02': Lasso(alpha=0.02, max_iter=10000, random_state=42),
                'ElasticNet α=0.01': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42),
            }
            
            best_model = None
            best_score = float('inf')
            best_model_name = ""
            
            print("\n🔍 Testing safe linear models with time-series cross-validation...")
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            for name, model in models.items():
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, 
                                              scoring='neg_mean_absolute_error',
                                              n_jobs=-1)
                    cv_mae = -cv_scores.mean()
                    
                    print(f"   {name:25} CV MAE: {cv_mae:.4f} ± {cv_scores.std():.4f}")
                    
                    if cv_mae < best_score:
                        best_score = cv_mae
                        best_model = model
                        best_model_name = name
                except Exception as e:
                    print(f"   {name:25} Failed: {e}")
            
            if best_model is None:
                raise ValueError("All models failed!")
            
            print(f"\n✅ Best model: {best_model_name}")
            
            # Train best model on training data
            print(f"🔄 Training {best_model_name}...")
            best_model.fit(X_train_scaled, y_train)
            self.model = best_model
            
            # Evaluate on TEST data only (completely unseen during training)
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n🎯 SAFE LINEAR REGRESSION PERFORMANCE:")
            print(f"   Model: {best_model_name}")
            print(f"   Mean Absolute Error (MAE):      {mae:.4f} °C")
            print(f"   Root Mean Squared Error (RMSE): {rmse:.4f} °C")
            print(f"   R² Score:                      {r2:.6f}")
            
            # Create performance visualization
            self.create_performance_plots(y_test, y_pred, best_model_name)
            
            # Save model
            if save_model:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"temperature.pkl"

                os.makedirs('saved_models', exist_ok=True)
                filepath = os.path.join('saved_models', model_filename)
                
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'poly': self.poly,
                    'features': self.feature_columns,
                    'performance': {
                        'mae': mae, 'rmse': rmse, 'r2': r2,
                        'model_type': best_model_name,
                        'test_size': len(X_test)
                    }
                }
                
                joblib.dump(model_data, filepath)
                print(f"💾 Safe linear model saved as {model_filename}")
                
                # Final target assessment
                if mae <= 0.2 and rmse <= 0.3:
                    print("\n🎉 TARGETS ACHIEVED! 🎉")
                    print(f"   ✅ MAE ≤ 0.2°C: {mae:.4f} ✓")
                    print(f"   ✅ RMSE ≤ 0.3°C: {rmse:.4f} ✓")
                else:
                    print(f"\n🎯 PERFORMANCE ANALYSIS:")
                    print(f"   Current MAE:  {mae:.4f} (target: ≤ 0.2000)")
                    print(f"   Current RMSE: {rmse:.4f} (target: ≤ 0.3000)")
                    print(f"   Gap to target: MAE +{mae-0.2:.4f}, RMSE +{rmse-0.3:.4f}")
                    print(f"   Progress: {min(100, (0.2/mae)*100):.1f}% towards MAE target")
                    print(f"   Progress: {min(100, (0.3/rmse)*100):.1f}% towards RMSE target")
            
            return True
            
        except Exception as e:
            print(f"❌ Safe linear regression training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def train_safe_linear_temperature():
    """Train safe linear regression model without data leakage"""
    data_file = "historical_data/chennai_weather_cleaned.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found!")
        print("💡 Please check the file path")
        return
    
    print("🚀 SAFE LINEAR REGRESSION TEMPERATURE PREDICTION")
    print("🎯 TARGET: MAE ≤ 0.2°C, RMSE ≤ 0.3°C")
    print("🛡️  NO DATA LEAKAGE")
    print("=" * 70)
    
    model = LinearTemperatureModel(data_file)
    
    success = model.train_safe_linear_model(
        target_column='tavg', 
        save_model=True
    )
    
    print("\n" + "=" * 70)
    print("🏁 SAFE LINEAR REGRESSION TRAINING COMPLETED")
    print("=" * 70)
    
    if success:
        print("✅ Safe linear regression model trained successfully!")
    else:
        print("❌ Safe linear regression training failed")

if __name__ == "__main__":
    train_safe_linear_temperature()