import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
import customtkinter as ctk
from tkinter import messagebox
import threading
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures

warnings.filterwarnings('ignore')

C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernWeatherApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Chennai AI Weather Forecast")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 700)
        
        self.colors = {
            'primary': '#3B82F6',
            'secondary': '#1E293B',
            'accent': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'dark_bg': '#0F172A',
            'card_bg': '#1E293B',
            'text_primary': '#F1F5F9',
            'text_secondary': '#94A3B8'
        }
        
        self.models = {}
        self.current_data = None
        self.forecast_days = 5
        self.model_features = {}
        self.model_preprocessors = {}
        
        self.window.configure(fg_color=self.colors['dark_bg'])
        
        self.setup_ui()
        self.load_models()
        self.load_current_data()
        
    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self.window, fg_color=self.colors['dark_bg'])
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.setup_header()
        
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, pady=10)
        
        self.content_frame.columnconfigure(1, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        
        self.setup_sidebar()
        self.setup_main_content()
        
    def setup_header(self):
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", height=60)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", fill="y", padx=25)
        
        ctk.CTkLabel(
            title_frame,
            text="Chennai AI Weather Forecast",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.colors['text_primary']
        ).pack(anchor="w")
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        ctk.CTkLabel(
            title_frame,
            text=current_date,
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_secondary']
        ).pack(anchor="w", pady=(2, 0))
        
    def setup_sidebar(self):
        sidebar = ctk.CTkFrame(self.content_frame, fg_color=self.colors['card_bg'], 
                              width=250, corner_radius=15)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        sidebar.pack_propagate(False)
        
        ctk.CTkLabel(
            sidebar,
            text="Forecast Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['text_primary']
        ).pack(anchor="w", padx=20, pady=20)
        
        days_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        days_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            days_frame,
            text="Forecast Period:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors['text_secondary']
        ).pack(anchor="w", pady=(0, 8))
        
        self.days_var = ctk.StringVar(value="5 Days")
        days_options = ["3 Days", "5 Days", "7 Days"]
        
        for option in days_options:
            radio = ctk.CTkRadioButton(
                days_frame,
                text=option,
                variable=self.days_var,
                value=option,
                command=self.on_days_change,
                font=ctk.CTkFont(size=12),
                text_color=self.colors['text_primary']
            )
            radio.pack(anchor="w", pady=3)
        
        button_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=15)
        
        self.forecast_btn = ctk.CTkButton(
            button_frame,
            text="Generate AI Forecast",
            command=self.generate_forecast,
            fg_color=self.colors['primary'],
            hover_color="#2563EB",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=40
        )
        self.forecast_btn.pack(fill="x", pady=(0, 8))
        
        self.refresh_btn = ctk.CTkButton(
            button_frame,
            text="Update Weather Data",
            command=self.update_weather_data,
            fg_color=self.colors['secondary'],
            hover_color="#374151",
            font=ctk.CTkFont(size=12),
            height=35
        )
        self.refresh_btn.pack(fill="x")
        
        status_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_frame.pack(fill="x", padx=20, pady=20, side="bottom")
        
        ctk.CTkLabel(
            status_frame,
            text="System Status",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.colors['text_primary']
        ).pack(anchor="w", pady=(0, 12))
        
        self.temp_model_status = self.create_status_indicator(
            status_frame, "Temperature Model", "Loading..."
        )
        self.precip_model_status = self.create_status_indicator(
            status_frame, "Precipitation Model", "Loading..."
        )
        self.data_status_label = self.create_status_indicator(
            status_frame, "Weather Data", "Loading..."
        )
        
    def create_status_indicator(self, parent, label, status):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(size=11),
            text_color=self.colors['text_secondary']
        ).pack(side="left")
        
        status_label = ctk.CTkLabel(
            frame,
            text=status,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=self.colors['warning']
        )
        status_label.pack(side="right")
        return status_label
        
    def setup_main_content(self):
        main_content = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        main_content.grid(row=0, column=1, sticky="nsew")
        
        self.setup_current_conditions(main_content)
        
        self.setup_forecast_display(main_content)
        
    def setup_current_conditions(self, parent):
        current_frame = ctk.CTkFrame(parent, fg_color=self.colors['card_bg'], 
                                   corner_radius=15, height=100)
        current_frame.pack(fill="x", pady=(0, 15))
        current_frame.pack_propagate(False)
        
        content = ctk.CTkFrame(current_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=25, pady=15)
        
        temp_frame = ctk.CTkFrame(content, fg_color="transparent")
        temp_frame.pack(side="left", fill="y")
        
        self.current_temp_label = ctk.CTkLabel(
            temp_frame,
            text="--\u00B0C",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.colors['text_primary']
        )
        self.current_temp_label.pack(anchor="w")
        
        self.current_weather_label = ctk.CTkLabel(
            temp_frame,
            text="Loading current weather...",
            font=ctk.CTkFont(size=13),
            text_color=self.colors['text_secondary']
        ).pack(anchor="w", pady=(3, 0))
        
        details_frame = ctk.CTkFrame(content, fg_color="transparent")
        details_frame.pack(side="right", fill="y")
        
        metrics_data = [
            ("Wind Speed", "current_wind", "-- km/h"), 
            ("Precipitation", "current_rain", "-- mm"),
            ("Pressure", "current_pressure", "-- hPa")
        ]
        
        for i, (icon, attr, default) in enumerate(metrics_data):
            metric_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
            metric_frame.pack(side="left", padx=(0 if i == 0 else 20))
            
            ctk.CTkLabel(
                metric_frame,
                text=icon,
                font=ctk.CTkFont(size=12),
                text_color=self.colors['text_secondary']
            ).pack(anchor="center")
            
            label_obj = ctk.CTkLabel(
                metric_frame,
                text=default,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=self.colors['text_primary']
            )
            label_obj.pack(anchor="center", pady=(2, 0))
            setattr(self, attr + "_label", label_obj)
        
    def setup_forecast_display(self, parent):
        forecast_frame = ctk.CTkFrame(parent, fg_color="transparent")
        forecast_frame.pack(fill="both", expand=True)
        
        title_frame = ctk.CTkFrame(forecast_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            title_frame,
            text="AI Weather Forecast",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['text_primary']
        ).pack(side="left")
        
        self.forecast_canvas = ctk.CTkScrollableFrame(
            forecast_frame,
            fg_color="transparent",
            scrollbar_button_color=self.colors['card_bg'],
            orientation="vertical"
        )
        self.forecast_canvas.pack(fill="both", expand=True)
        
        self.forecast_cards_container = ctk.CTkFrame(self.forecast_canvas, fg_color="transparent")
        self.forecast_cards_container.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.show_welcome_message()
        
    def show_welcome_message(self):
        for widget in self.forecast_cards_container.winfo_children():
            widget.destroy()
        
        welcome_frame = ctk.CTkFrame(self.forecast_cards_container, fg_color="transparent")
        welcome_frame.pack(expand=True, fill="both", pady=50)
        
        ctk.CTkLabel(
            welcome_frame,
            text="\u2600",
            font=ctk.CTkFont(size=48),
            text_color=self.colors['text_secondary']
        ).pack(pady=(0, 20))
        
        ctk.CTkLabel(
            welcome_frame,
            text="Chennai AI Weather Forecast",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['text_primary']
        ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            welcome_frame,
            text="Select forecast days and click 'Generate AI Forecast'\nto see AI-powered weather predictions",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_secondary'],
            justify="center"
        ).pack()
        
    def load_models(self):
        self.models = {}
        model_files = {
            'temperature': 'saved_models/temperature.pkl',
            'precipitation': 'saved_models/precipitator.pkl'
        }
        
        precip_files = [
            'saved_models/precipitator.joblib', 
        ]
        
        temp_loaded = False
        if os.path.exists(model_files['temperature']):
            try:
                model_data = joblib.load(model_files['temperature'])
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.models['temperature'] = model_data['model']
                    self.model_features['temperature'] = model_data.get('features', [])
                    self.model_preprocessors['temperature'] = {
                        'scaler': model_data.get('scaler'),
                        'poly': model_data.get('poly')
                    }
                    print(f"Loaded temperature model with {len(self.model_features['temperature'])} features")
                    self.temp_model_status.configure(text="Loaded", text_color=self.colors['accent'])
                    temp_loaded = True
            except Exception as e:
                print(f"{C_RED}Error loading temperature model: {e}{C_RESET}")
        
        if not temp_loaded:
            self.temp_model_status.configure(text="Not found", text_color=self.colors['danger'])
        
        precip_loaded = False
        self.model_preprocessors['precipitation'] = {}
        for precip_file in precip_files:
            if os.path.exists(precip_file):
                try:
                    print(f"Trying to load precipitation model from: {precip_file}")
                    model_data = joblib.load(precip_file)
                    
                    if isinstance(model_data, dict):
                        if 'models' in model_data and model_data['models']:
                            self.models['precipitation'] = model_data['models']
                            self.model_features['precipitation'] = model_data.get('feature_columns', [])
                            self.model_preprocessors['precipitation']['scaler'] = model_data.get('scaler')
                            print(f"Loaded precipitation ensemble model with {len(self.model_features['precipitation'])} features")
                            precip_loaded = True
                            break
                        elif 'model' in model_data:
                            self.models['precipitation'] = [model_data['model']]
                            self.model_features['precipitation'] = model_data.get('features', [])
                            self.model_preprocessors['precipitation']['scaler'] = model_data.get('scaler')
                            print(f"Loaded precipitation single model")
                            precip_loaded = True
                            break
                except Exception as e:
                    print(f"{C_RED}Error loading precipitation model from {precip_file}: {e}{C_RESET}")
        
        if not precip_loaded:
            self.precip_model_status.configure(text="Not found", text_color=self.colors['danger'])
        else:
            self.precip_model_status.configure(text="Loaded", text_color=self.colors['accent'])
        
    def load_current_data(self):
        try:
            data_files = [
                'historical_data/chennai_weather_cleaned.csv',
                'recent_weather.csv',
                'chennai_weather_data.csv'
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    self.current_data = pd.read_csv(file_path)
                    
                    if 'date' in self.current_data.columns:
                        self.current_data['date'] = pd.to_datetime(self.current_data['date'])
                        self.current_data = self.current_data.sort_values('date').reset_index(drop=True)
                    
                    print(f"Loaded {len(self.current_data)} rows from {file_path}")
                    self.update_current_display()
                    self.data_status_label.configure(text="Loaded", text_color=self.colors['accent'])
                    return
            
            self.data_status_label.configure(text="No files", text_color=self.colors['danger'])
        except Exception as e:
            print(f"{C_RED}Data loading error: {e}{C_RESET}")
            self.data_status_label.configure(text="Error", text_color=self.colors['danger'])
    
    def update_current_display(self):
        if self.current_data is not None and not self.current_data.empty:
            latest = self.current_data.iloc[-1]
            
            temp_value = None
            for temp_col in ['tavg', 'temp', 'temperature', 'tmax']:
                if temp_col in latest and pd.notna(latest[temp_col]):
                    temp_value = float(latest[temp_col])
                    break
            
            if temp_value is not None:
                self.current_temp_label.configure(text=f"{temp_value:.1f}\u00B0C")
                if temp_value > 35: weather_desc = "Very Hot"
                elif temp_value > 30: weather_desc = "Hot"
                elif temp_value > 25: weather_desc = "Warm"
                elif temp_value > 20: weather_desc = "Mild"
                else: weather_desc = "Cool"
                    
                latest_prcp = latest.get('prcp', 0.0)
                if latest_prcp > 5.0: weather_desc = "Rainy"
                elif latest_prcp > 0.1: weather_desc = "Light Rain"
                    
                self.current_weather_label.configure(text=weather_desc)
            
            metric_mapping = {
                'wspd': ('current_wind', '{:.1f} km/h'),
                'prcp': ('current_rain', '{:.1f} mm'),
                'pres': ('current_pressure', '{:.0f} hPa')
            }
            
            for data_col, (label_attr, format_str) in metric_mapping.items():
                if data_col in latest and pd.notna(latest[data_col]):
                    label = getattr(self, f"{label_attr}_label")
                    label.configure(text=format_str.format(float(latest[data_col])))
    
    def on_days_change(self):
        days_text = self.days_var.get()
        if days_text == "3 Days": self.forecast_days = 3
        elif days_text == "5 Days": self.forecast_days = 5
        elif days_text == "7 Days": self.forecast_days = 7
    
    def update_weather_data(self):
        def update_thread():
            self.refresh_btn.configure(state="disabled", text="Updating...")
            try:
                try:
                    from get_fresh_data import json_to_csv
                    result = json_to_csv()
                    if result: messagebox.showinfo("Success", "Weather data updated successfully!")
                    else: messagebox.showwarning("Warning", "Data update completed but no data was returned")
                except ImportError:
                    messagebox.showwarning("Info", "Data update module not available")
                self.load_current_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update data: {str(e)}")
            finally:
                self.refresh_btn.configure(state="normal", text="Update Weather Data")
        threading.Thread(target=update_thread, daemon=True).start()
    
    def generate_forecast(self):
        def forecast_thread():
            self.forecast_btn.configure(state="disabled", text="Generating...")
            try:
                if not self.models:
                    messagebox.showerror("Error", "No AI models loaded. Please check model files.")
                    return
                if self.current_data is None or self.current_data.empty:
                    messagebox.showwarning("Warning", "No weather data available.")
                    return
                
                print(f"\n{C_BLUE}--- Starting AI Forecast Generation ---{C_RESET}")
                
                for widget in self.forecast_cards_container.winfo_children():
                    widget.destroy()
                
                predictions = self.generate_ai_predictions()
                
                if not predictions:
                    messagebox.showerror("Error", "Failed to generate predictions.")
                    return
                
                print(f"{C_GREEN}Successfully generated {len(predictions)} weather predictions using AI{C_RESET}")
                self.display_forecast_cards(predictions)
                messagebox.showinfo("Success", f"Generated {len(predictions)}-day AI forecast!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Forecast failed: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                self.forecast_btn.configure(state="normal", text="Generate AI Forecast")
        threading.Thread(target=forecast_thread, daemon=True).start()
    
    def generate_ai_predictions(self):
        predictions = []
        
        df_full = self.current_data.copy()
        df_full['date'] = pd.to_datetime(df_full['date'])
        
        for col in ['tavg', 'prcp', 'tmin', 'tmax', 'pres', 'wspd']:
            if col not in df_full.columns:
                if col == 'pres': df_full[col] = 1010.0
                elif col == 'wspd': df_full[col] = 10.0
                elif col == 'prcp': df_full[col] = 0.0
                else: df_full[col] = df_full[col].mean() if not df_full.empty else 28.0 
        
        latest_date = df_full['date'].max()
        
        for day in range(1, self.forecast_days + 1):
            prediction_date = latest_date + timedelta(days=day)
            
            new_row_data = {
                'date': prediction_date,
                'year': prediction_date.year,
                'month': prediction_date.month,
                'day': prediction_date.day,
                'day_of_year': prediction_date.timetuple().tm_yday,
                'day_of_week': prediction_date.dayofweek,
                'tavg': np.nan, 'prcp': np.nan
            }
            df_full = pd.concat([df_full, pd.DataFrame([new_row_data])], ignore_index=True)
            
            df_features = self.recreate_features(df_full.copy(), prediction_date)
            X_pred_day = df_features[df_features['date'] == prediction_date]
            
            if X_pred_day.empty:
                print(f"{C_RED}Error: Could not create features for {prediction_date.date()}{C_RESET}")
                continue
            
            temp_pred = self.predict_temperature_ai(X_pred_day)
            is_temp_ai = True
            if temp_pred is None: 
                temp_pred = 0.0
                is_temp_ai = False
            
            precip_pred_mm = self.predict_precipitation_ai(X_pred_day)
            is_precip_ai = True
            if precip_pred_mm is None: 
                precip_pred_mm = 0.0
                is_precip_ai = False
            
            current_history = df_full[df_full['date'] < prediction_date]
            temp_pred, precip_pred_mm, weather_override = self.detect_and_adjust_for_cyclone(
                current_history, prediction_date, temp_pred, precip_pred_mm
            )
            
            df_full.loc[df_full['date'] == prediction_date, 'tavg'] = temp_pred
            df_full.loc[df_full['date'] == prediction_date, 'prcp'] = precip_pred_mm
            
            rain_chance = self.mm_to_rain_chance(precip_pred_mm)
            
            if weather_override:
                weather_desc = weather_override
                icon = "\u25CF" 
                color_code = C_RED if "CYCLONE" in weather_override else C_YELLOW
            else:
                weather_desc, icon = self.get_weather_condition(temp_pred, rain_chance, precip_pred_mm)
                color_code = C_RESET

            print(f"{color_code}Day {day}: {weather_desc} | Temp: {temp_pred:.1f}C | Precip: {precip_pred_mm:.1f}mm{C_RESET}")
            
            is_successful_ai = is_temp_ai and is_precip_ai
            
            prediction_data = {
                'date': prediction_date.strftime('%a, %b %d'),
                'day_name': prediction_date.strftime('%A'),
                'temp': round(temp_pred, 1),
                'weather': weather_desc,
                'icon': icon,
                'rain_chance': rain_chance,
                'precip_mm': round(precip_pred_mm, 1),
                'is_ai': is_successful_ai,
                'temp_color': self.get_temperature_color(temp_pred)
            }
            predictions.append(prediction_data)
        
        return predictions

    def recreate_features(self, df, prediction_date):
        df = df.sort_values('date').reset_index(drop=True)
        df = df[df['date'] <= prediction_date].copy()
        
        df['day_of_year_rad'] = 2 * np.pi * df['day_of_year'] / 365.25
        for h in [1, 2, 3]:
            df[f'day_sin_{h}'] = np.sin(h * df['day_of_year_rad'])
            df[f'day_cos_{h}'] = np.cos(h * df['day_of_year_rad'])
            
        df['month_rad'] = 2 * np.pi * df['month'] / 12
        df['month_sin'] = np.sin(df['month_rad'])
        df['month_cos'] = np.cos(df['month_rad'])
        df['month_sin_2'] = np.sin(2 * df['month_rad'])
        
        if 'tmin' in df.columns and 'tmax' in df.columns:
            df['temp_range'] = df['tmax'] - df['tmin']
            df['temp_avg_min_max'] = (df['tmin'] + df['tmax']) / 2
        
        for col in ['tavg', 'prcp', 'pres', 'wspd']:
            if col in df.columns:
                for lag in [1, 2, 3, 4, 7, 14]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        df['year_trend'] = df['year'] - df['year'].min()
        df['year_trend_scaled'] = df['year_trend'] / df['year_trend'].max()
        
        df['temp_change_1d'] = df['tavg_lag_1'] - df['tavg_lag_2']
        df['temp_change_2d'] = df['tavg_lag_2'] - df['tavg_lag_3']
        df['temp_volatility'] = df[['temp_change_1d', 'temp_change_2d']].std(axis=1)

        for col in ['tavg', 'prcp']:
            if col in df.columns:
                for w in [7, 14]:
                    df[f'{col}_expanding_mean_{w}'] = df[col].expanding(min_periods=w).mean().shift(1)
                    df[f'{col}_expanding_std_{w}'] = df[col].expanding(min_periods=w).std().shift(1)
                    
                for w in [3, 7, 14]:
                    df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean().shift(1)

        last_idx = df.index[-1]
        for col in df.columns:
            if pd.isna(df.loc[last_idx, col]) and col not in ['tavg', 'prcp']: 
                if last_idx > 0:
                    df.loc[last_idx, col] = df.loc[last_idx-1, col]
                else:
                    df.loc[last_idx, col] = 0
                    
        return df

    def predict_temperature_ai(self, X_pred_day):
        if 'temperature' not in self.models: return None

        try:
            model = self.models['temperature']
            poly = self.model_preprocessors['temperature'].get('poly')
            scaler = self.model_preprocessors['temperature'].get('scaler')
            
            if hasattr(poly, 'feature_names_in_'):
                expected_features = poly.feature_names_in_
            else:
                expected_features = self.model_features['temperature']
            
            for f in expected_features:
                if f not in X_pred_day.columns:
                    print(f"{C_RED}Missing required feature: {f} for Temp Model. Imputing with 0.{C_RESET}")
                    X_pred_day[f] = 0

            X_filtered = X_pred_day[expected_features].copy()
            
            X_input = X_filtered.values
            if poly: X_input = poly.transform(X_input)
            if scaler: X_input = scaler.transform(X_input)
            
            pred = model.predict(X_input)[0]
            return float(pred)
        except Exception as e:
            print(f"{C_RED}FATAL AI ERROR (Temp): {type(e).__name__}: {e}{C_RESET}")
            return None

    def predict_precipitation_ai(self, X_pred_day):
        if 'precipitation' not in self.models: return None
        
        try:
            ensemble = self.models['precipitation']
            expected_features = self.model_features.get('precipitation', [])
            
            if not expected_features: return 0.5
            
            for f in expected_features:
                if f not in X_pred_day.columns:
                    print(f"{C_RED}Missing required feature: {f} for Precip Model. Imputing with 0.{C_RESET}")
                    X_pred_day[f] = 0
            
            X_filtered = X_pred_day[expected_features].copy()
            
            scaler = self.model_preprocessors['precipitation'].get('scaler')
            X_input = X_filtered.values
            if scaler: X_input = scaler.transform(X_input)
            
            preds = []
            if isinstance(ensemble, list):
                for m in ensemble: preds.append(m.predict(X_input)[0])
                final_pred = np.median(preds)
            else:
                final_pred = ensemble.predict(X_input)[0]
                
            return float(max(0.0, final_pred))
        except Exception as e:
            print(f"{C_RED}FATAL AI ERROR (Precip): {type(e).__name__}: {e}{C_RESET}")
            return None

    def detect_and_adjust_for_cyclone(self, history_df, prediction_date, temp_pred, precip_pred_mm):
        if history_df.empty: return temp_pred, precip_pred_mm, None

        latest = history_df.iloc[-1]
        pressure = latest.get('pres', 1010.0)
        wind = latest.get('wspd', 10.0)
        
        prev_pres = history_df.iloc[-2].get('pres', 1010.0) if len(history_df) > 1 else pressure
        pres_drop = prev_pres - pressure

        month = prediction_date.month
        is_season = month in [10, 11, 12, 4, 5]
        
        override = None
        
        is_depression = (wind > 31 and pressure < 1004) or (pressure < 1000) or (pres_drop > 3 and pressure < 1006)
        is_cyclone = (wind > 62) or (pressure < 990)

        if is_season:
            if is_cyclone:
                print(f"{C_RED}CYCLONE ALERT DETECTED: {prediction_date.date()}{C_RESET}")
                precip_pred_mm = max(50.0, precip_pred_mm * 3.0)
                temp_pred = min(temp_pred, 25.0)
                override = "CYCLONE ALERT"
            elif is_depression:
                print(f"{C_YELLOW}DEPRESSION DETECTED: {prediction_date.date()}{C_RESET}")
                precip_pred_mm = max(20.0, precip_pred_mm * 1.5)
                temp_pred -= 2.0
                override = "Depression"
            elif pressure < 1006:
                precip_pred_mm = max(10.0, precip_pred_mm * 1.2)
                override = "Low Pressure"

        return temp_pred, precip_pred_mm, override

    def mm_to_rain_chance(self, mm):
        if mm < 0.1: return 0
        chance = 25.0 * np.log(mm + 1)
        return int(np.clip(chance, 0, 100))

    def get_weather_condition(self, temp, rain_chance, precip_mm):
        if rain_chance > 70 or precip_mm > 10:
            return "Heavy Rain" if precip_mm > 20 else "Rain", "\u2614"
        elif rain_chance > 40: return "Light Rain", "\u2601"
        elif temp > 35: return "Very Hot", "\u2600"
        elif temp > 30: return "Hot", "\u2600"
        elif temp > 25: return "Warm", "\u2600"
        else: return "Cool", "\u2601"
    
    def get_temperature_color(self, temp):
        if temp > 35: return "#EF4444"
        elif temp > 30: return "#F59E0B"
        elif temp > 25: return "#10B981"
        else: return "#3B82F6"
    
    def display_forecast_cards(self, predictions):
        for widget in self.forecast_cards_container.winfo_children(): widget.destroy()
        
        if not predictions:
            self.show_welcome_message()
            return
            
        cards_container = ctk.CTkFrame(self.forecast_cards_container, fg_color="transparent")
        cards_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        for pred in predictions:
            self.create_forecast_card(pred).pack(fill="x", pady=8, padx=5)

    def create_forecast_card(self, prediction):
        card = ctk.CTkFrame(self.forecast_cards_container, fg_color=self.colors['card_bg'], corner_radius=15, height=120)
        card.pack_propagate(False)
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=15)
        
        left = ctk.CTkFrame(content, fg_color="transparent")
        left.pack(side="left", fill="y", padx=(0, 20))
        ctk.CTkLabel(left, text=prediction['date'], font=ctk.CTkFont(size=16, weight="bold"), text_color=self.colors['text_primary']).pack(anchor="w")
        ctk.CTkLabel(left, text=prediction['day_name'], font=ctk.CTkFont(size=13), text_color=self.colors['text_secondary']).pack(anchor="w")
        
        if prediction.get('is_ai', True):
            badge = ctk.CTkFrame(left, fg_color=self.colors['accent'], corner_radius=8)
            badge.pack(anchor="w", pady=(8, 0))
            ctk.CTkLabel(badge, text="AI PREDICTION", font=ctk.CTkFont(size=10, weight="bold"), text_color="white").pack(padx=8, pady=3)
        else:
            badge = ctk.CTkFrame(left, fg_color=self.colors['danger'], corner_radius=8)
            badge.pack(anchor="w", pady=(8, 0))
            ctk.CTkLabel(badge, text="AI FAILED", font=ctk.CTkFont(size=10, weight="bold"), text_color="white").pack(padx=8, pady=3)
        
        center = ctk.CTkFrame(content, fg_color="transparent")
        center.pack(side="left", fill="y", padx=20)
        ctk.CTkLabel(center, text=prediction['icon'], font=ctk.CTkFont(size=28), text_color=self.colors['text_primary']).pack(anchor="center")
        ctk.CTkLabel(center, text=prediction['weather'], font=ctk.CTkFont(size=14, weight="bold"), text_color=self.colors['text_primary']).pack(anchor="center")
        
        right = ctk.CTkFrame(content, fg_color="transparent")
        right.pack(side="right", fill="y")
        ctk.CTkLabel(right, text=f"{prediction['temp']}\u00B0C", font=ctk.CTkFont(size=22, weight="bold"), text_color=prediction['temp_color']).pack(anchor="e")
        ctk.CTkLabel(right, text=f"\u2602 {prediction['rain_chance']}% chance", font=ctk.CTkFont(size=12, weight="bold"), text_color="#60A5FA").pack(anchor="e")
        ctk.CTkLabel(right, text=f"{prediction['precip_mm']}mm expected", font=ctk.CTkFont(size=11), text_color="#60A5FA").pack(anchor="e")
        
        return card

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = ModernWeatherApp()
    app.run()