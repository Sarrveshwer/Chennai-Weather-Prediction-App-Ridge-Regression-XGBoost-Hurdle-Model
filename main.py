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
import requests
from bs4 import BeautifulSoup
import random

warnings.filterwarnings('ignore')

C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def check_imd_status():
    url = "https://mausam.imd.gov.in/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        page_text = soup.get_text().lower()
        
        found_system = False
        warning_msg = ""

        keywords = ["cyclone", "depression", "deep depression", "severe storm", "fengal", "low pressure"]
        locations = ["tamil nadu", "chennai", "bay of bengal", "coast", "puducherry", "south india"]

        for k in keywords:
            if k in page_text:
                if any(l in page_text for l in locations):
                    found_system = True
                    warning_msg = f"{k.upper()} Alert Detected near Chennai/TN"
                    break
        
        if found_system:
            return True, warning_msg
        else:
            return False, ""

    except Exception:
        return False, ""

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
        self.imd_alert_active = False
        self.imd_alert_message = ""
        
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
            text="--°C",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.colors['text_primary']
        )
        self.current_temp_label.pack(anchor="w")
        
        self.current_weather_label = ctk.CTkLabel(
            temp_frame,
            text="Loading current weather...",
            font=ctk.CTkFont(size=13),
            text_color=self.colors['text_secondary']
        )
        self.current_weather_label.pack(anchor="w", pady=(3, 0))
        
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
                if hasattr(self, 'current_temp_label'):
                    self.current_temp_label.configure(text=f"{temp_value:.1f}°C")
                
                if temp_value > 35: weather_desc = "Very Hot"
                elif temp_value > 30: weather_desc = "Hot"
                elif temp_value > 25: weather_desc = "Warm"
                elif temp_value > 20: weather_desc = "Mild"
                else: weather_desc = "Cool"
                    
                latest_prcp = latest.get('prcp', 0.0)
                if latest_prcp > 5.0: weather_desc = "Rainy"
                elif latest_prcp > 0.1: weather_desc = "Light Rain"
                    
                if hasattr(self, 'current_weather_label'):
                    self.current_weather_label.configure(text=weather_desc)
            
            metric_mapping = {
                'wspd': ('current_wind', '{:.1f} km/h'),
                'prcp': ('current_rain', '{:.1f} mm'),
                'pres': ('current_pressure', '{:.0f} hPa')
            }
            
            for data_col, (label_attr, format_str) in metric_mapping.items():
                if data_col in latest and pd.notna(latest[data_col]):
                    label_name = f"{label_attr}_label"
                    if hasattr(self, label_name):
                        label = getattr(self, label_name)
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
                alert_active, alert_msg = check_imd_status()
                if alert_active:
                    self.imd_alert_active = True
                    self.imd_alert_message = alert_msg
                    self.window.after(0, lambda: messagebox.showwarning("CRITICAL WEATHER ALERT", f"IMD ALERT DETECTED:\n{alert_msg}"))
                else:
                    self.imd_alert_active = False
                    
                try:
                    from get_fresh_data import json_to_csv
                    # Correctly handle tuple return from json_to_csv
                    result = json_to_csv()
                    
                    # Check if we got valid data
                    if result is not None:
                        if isinstance(result, tuple):
                            df = result[0]
                        else:
                            df = result
                            
                        if df is not None and not df.empty:
                            self.window.after(0, lambda: messagebox.showinfo("Success", "Weather data updated successfully!"))
                        else:
                            self.window.after(0, lambda: messagebox.showwarning("Warning", "Data update completed but no data was returned"))
                    else:
                        self.window.after(0, lambda: messagebox.showwarning("Warning", "Data update failed"))
                        
                except ImportError:
                    self.window.after(0, lambda: messagebox.showwarning("Info", "Data update module not available"))
                self.load_current_data()
            except Exception as e:
                error_msg = str(e)
                self.window.after(0, lambda: messagebox.showerror("Error", f"Failed to update data: {error_msg}"))
            finally:
                self.window.after(0, lambda: self.refresh_btn.configure(state="normal", text="Update Weather Data"))
        threading.Thread(target=update_thread, daemon=True).start()
    
    def generate_forecast(self):
        def forecast_thread():
            self.forecast_btn.configure(state="disabled", text="Generating...")
            try:
                if not self.models:
                    self.window.after(0, lambda: messagebox.showerror("Error", "No AI models loaded. Please check model files."))
                    return
                if self.current_data is None or self.current_data.empty:
                    self.window.after(0, lambda: messagebox.showwarning("Warning", "No weather data available."))
                    return
                
                print(f"\n{C_BLUE}--- Starting AI Forecast Generation ---{C_RESET}")
                
                alert_active, alert_msg = check_imd_status()
                if alert_active:
                    self.imd_alert_active = True
                    self.imd_alert_message = alert_msg
                    print(f"{C_RED}IMD ALERT FOUND: {alert_msg}{C_RESET}")
                
                predictions = self.generate_ai_predictions()
                
                if not predictions:
                    self.window.after(0, lambda: messagebox.showerror("Error", "Failed to generate predictions."))
                    return
                
                print(f"{C_GREEN}Successfully generated {len(predictions)} weather predictions using AI{C_RESET}")
                self.window.after(0, lambda: self.display_forecast_cards(predictions))
                self.window.after(0, lambda: messagebox.showinfo("Success", f"Generated {len(predictions)}-day AI forecast!"))
                
            except Exception as e:
                error_msg = str(e)
                self.window.after(0, lambda: messagebox.showerror("Error", f"Forecast failed: {error_msg}"))
                import traceback
                traceback.print_exc()
            finally:
                self.window.after(0, lambda: self.forecast_btn.configure(state="normal", text="Generate AI Forecast"))
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
        
        current_real_date = datetime.now()
        
        for day in range(1, self.forecast_days + 1):
            prediction_date = current_real_date + timedelta(days=day)
            
            last_row = df_full.iloc[-1]
            
            cyclone_pres = last_row['pres']
            cyclone_wind = last_row['wspd']
            
            if self.imd_alert_active:
                if day == 1:
                    cyclone_pres = min(cyclone_pres, 1000.0)
                    cyclone_wind = max(cyclone_wind, 40.0)
                else:
                    cyclone_pres = min(cyclone_pres - 5, 990.0)
                    cyclone_wind = max(cyclone_wind + 15, 80.0)

            new_row_data = {
                'date': prediction_date,
                'year': prediction_date.year,
                'month': prediction_date.month,
                'day': prediction_date.day,
                'day_of_year': prediction_date.timetuple().tm_yday,
                'day_of_week': prediction_date.weekday(),
                'tavg': last_row['tavg'],
                'prcp': 0.0,
                'tmin': last_row['tmin'],
                'tmax': last_row['tmax'],
                'pres': cyclone_pres,
                'wspd': cyclone_wind
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
                current_history, prediction_date, temp_pred, precip_pred_mm, day
            )
            
            idx = df_full.index[-1]
            df_full.loc[idx, 'tavg'] = temp_pred
            df_full.loc[idx, 'prcp'] = precip_pred_mm
            df_full.loc[idx, 'tmin'] = temp_pred - 4
            df_full.loc[idx, 'tmax'] = temp_pred + 4
            
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
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['doy_sin'] = df['day_sin_1']
        df['doy_cos'] = df['day_cos_1']

        def seasonal_weight(month):
            if month in [12, 1]: return 1.0
            elif month in [11, 2]: return 0.8
            elif month in [10, 3]: return 0.5
            elif month in [9, 4]: return 0.3
            else: return 0.0
            
        df['winter_strength'] = df['month'].apply(seasonal_weight)
        df['monsoon_strength'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(float) * 0.8
        df['summer_strength'] = ((df['month'] >= 3) & (df['month'] <= 5)).astype(float) * 0.6
        df['is_monsoon'] = (df['month'].isin([6, 7, 8, 9, 10])).astype(int)
        df['is_summer'] = (df['month'].isin([3, 4, 5])).astype(int)
        df['is_winter'] = (df['month'].isin([11, 12, 1, 2])).astype(int)

        if 'tmin' in df.columns and 'tmax' in df.columns:
            df['temp_range'] = df['tmax'] - df['tmin']
            df['temp_avg_min_max'] = (df['tmin'] + df['tmax']) / 2
            df['temp_center'] = df['tmin'] + (df['temp_range'] / 2)
        
        monthly_tavg = df.groupby('month')['tavg'].transform('mean')
        df['temp_anomaly'] = df['tavg'] - monthly_tavg
        if 'pres' in df.columns:
            monthly_pres = df.groupby('month')['pres'].transform('mean')
            df['pressure_anomaly'] = df['pres'] - monthly_pres
        
        for col in ['tavg', 'prcp', 'pres', 'wspd']:
            if col in df.columns:
                for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        for lag in [1, 2, 3, 7, 14]:
            df[f'had_rain_lag_{lag}'] = (df[f'prcp_lag_{lag}'] > 0.1).astype(int)

        df['year_trend'] = df['year'] - df['year'].min()
        df['year_trend_scaled'] = df['year_trend'] / (df['year_trend'].max() + 1e-5)
        df['year_trend_squared'] = df['year_trend'] ** 2
        
        df['temp_change_1d'] = df['tavg_lag_1'] - df['tavg_lag_2']
        df['temp_change_2d'] = df['tavg_lag_2'] - df['tavg_lag_3']
        df['temp_change_3d'] = df['tavg_lag_3'] - df['tavg_lag_4']
        df['temp_change_magnitude'] = np.abs(df['temp_change_1d'])
        df['temp_volatility'] = df[['temp_change_1d', 'temp_change_2d']].std(axis=1)
        df['temp_momentum'] = df['temp_change_1d'] - df['temp_change_2d']
        
        for col in ['tavg', 'prcp', 'pres', 'wspd']:
            if col in df.columns:
                for w in [3, 7, 14, 30, 60]:
                    df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w, min_periods=1).mean().shift(1)
                    df[f'{col}_rolling_std_{w}'] = df[col].rolling(w, min_periods=1).std().shift(1)
                    if col == 'prcp':
                        df[f'{col}_rolling_sum_{w}'] = df[col].rolling(w, min_periods=1).sum().shift(1)
                        df[f'{col}_max_{w}'] = df[col].rolling(w).max()
                
                for d in [1, 3, 7]:
                    name = 'temp' if col == 'tavg' else 'pressure' if col == 'pres' else 'wind' if col == 'wspd' else col
                    df[f'{name}_trend_{d}d'] = df[col].diff(d).shift(1)

        for w in [7, 14]:
            df[f'tavg_expanding_mean_{w}'] = df['tavg'].expanding(min_periods=w).mean().shift(1)
            df[f'tavg_expanding_std_{w}'] = df['tavg'].expanding(min_periods=w).std().shift(1)

        df['prcp_sqrt'] = np.sqrt(df['prcp_lag_1'] + 1)
        df['had_precipitation'] = (df['prcp_lag_1'] > 0).astype(int)
        df['heavy_rain'] = (df['prcp_lag_1'] > 10).astype(int)
        df['recent_rain'] = df['had_rain_lag_1']
        
        for span in [7, 14, 30]:
            df[f'prcp_ema_{span}'] = df['prcp'].ewm(span=span, adjust=False).mean().shift(1)
            
        df['prcp_skew_7d'] = df['prcp'].rolling(7).skew().shift(1)
        df['prcp_kurt_30'] = df['prcp'].rolling(30).kurt().shift(1)
        df['prcp_skew_30'] = df['prcp'].rolling(30).skew().shift(1)
        
        df['winter_temp_effect'] = df['winter_strength'] * df['tavg_lag_1']
        df['monsoon_temp_effect'] = df['monsoon_strength'] * df['tavg_lag_1']
        df['precip_cooling_effect'] = df['prcp_sqrt'] * df['monsoon_strength']
        df['temp_range_seasonal'] = df['temp_range'] * df['summer_strength']
        df['lag_interaction'] = df['tavg_lag_1'] * df['tavg_lag_2']
        
        if 'pres' in df.columns and 'wspd' in df.columns:
            df['temp_pressure'] = df['tavg'] * df['pres']
            df['temp_wind'] = df['tavg'] * df['wspd']
            df['pressure_wind'] = df['pres'] * df['wspd']
            df['monsoon_temp'] = df['is_monsoon'] * df['tavg']
            df['monsoon_pressure'] = df['is_monsoon'] * df['pres']
            df['monsoon_wind'] = df['is_monsoon'] * df['wspd']
            df['temp_rain_interaction'] = df['tavg'] * df['prcp_lag_1']
            df['pressure_rain_interaction'] = df['pres'] * df['prcp_lag_1']
            df['weather_regime'] = pd.cut(df['tavg'], bins=5, labels=False) * 10 + pd.cut(df['pres'], bins=5, labels=False)

        for w in [7, 14, 30]:
            df[f'prcp_rolling_q25_{w}'] = df['prcp'].rolling(w).quantile(0.25).shift(1)
            df[f'prcp_rolling_q75_{w}'] = df['prcp'].rolling(w).quantile(0.75).shift(1)
            df[f'prcp_rolling_iqr_{w}'] = df[f'prcp_rolling_q75_{w}'] - df[f'prcp_rolling_q25_{w}']

        df['seasonal_adjustment'] = df['tavg'] - monthly_tavg
        df['year_frac'] = (df['day_of_year'] - 1) / 365.0
        df['seasonal_component'] = np.sin(2 * np.pi * df['year_frac']) + np.cos(4 * np.pi * df['year_frac'])
        df['prcp_cumulative_30d'] = df['prcp_rolling_sum_30']
        monthly_cum_prcp = df.groupby('month')['prcp_cumulative_30d'].transform('mean')
        df['prcp_deficit'] = df['prcp_cumulative_30d'] - monthly_cum_prcp
        
        rain_mask = df['prcp'] > 0.1
        df['consecutive_rain_days'] = rain_mask.groupby((rain_mask != rain_mask.shift()).cumsum()).cumsum()
        dry_mask = df['prcp'] <= 0.1
        df['consecutive_dry_days'] = dry_mask.groupby((dry_mask != dry_mask.shift()).cumsum()).cumsum()
        
        df['prcp_winsorized'] = df['prcp'].clip(upper=df['prcp'].quantile(0.95))
        
        if 'tavg' in df.columns and 'pres' in df.columns:
            df['temp_pres_interaction'] = df['tavg'] * np.log1p(df['pres'])
        
        if 'wspd' in df.columns and 'pres' in df.columns:
            df['storm_energy'] = (df['wspd'] ** 2) * (1013 - df['pres'])

        # Log-transform highly skewed features
        if 'wspd' in df.columns: df['wspd_log'] = np.log1p(df['wspd'])
        if 'pres' in df.columns: df['pres_log'] = np.log1p(df['pres'])

        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        df = df.fillna(0)

        return df.iloc[[-1]]

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
            
            if isinstance(ensemble, dict) and 'models' in ensemble:
                models = ensemble['models']
                expected_features = ensemble['feature_columns']
            else:
                return 0.0
            
            for f in expected_features:
                if f not in X_pred_day.columns:
                    print(f"{C_RED}Missing required feature: {f} for Precip Model. Imputing with 0.{C_RESET}")
                    X_pred_day[f] = 0
            
            X_filtered = X_pred_day[expected_features].copy()
            
            scaler = self.model_preprocessors['precipitation'].get('scaler')
            X_input = X_filtered.values
            if scaler: X_input = scaler.transform(X_input)
            
            preds = []
            for m in models:
                pred = m.predict(X_input)[0]
                preds.append(pred)
            
            final_pred = (0.4 * preds[0]) + (0.3 * preds[1]) + (0.3 * preds[2])
            
            return float(max(0.0, final_pred))
            
        except Exception as e:
            print(f"{C_RED}FATAL AI ERROR (Precip): {type(e).__name__}: {e}{C_RESET}")
            return None

    def detect_and_adjust_for_cyclone(self, history_df, prediction_date, temp_pred, precip_pred_mm, day_offset):
        if history_df.empty: return temp_pred, precip_pred_mm, None

        override = None
        
        if self.imd_alert_active:
            print(f"{C_RED}EXTERNAL IMD ALERT ACTIVE (Day {day_offset}){C_RESET}")
            
            if day_offset <= 2:
                rain_boost = 40.0 + (day_offset * 10.0) + random.uniform(-5, 5)
                precip_pred_mm = max(rain_boost, precip_pred_mm * 3.0)
                temp_pred = min(temp_pred, 25.0)
                override = f"⚠ {self.imd_alert_message[:20]}..."
            elif day_offset == 3:
                rain_boost = 80.0 + random.uniform(-10, 10)
                precip_pred_mm = max(rain_boost, precip_pred_mm * 4.0)
                temp_pred = min(temp_pred, 24.0)
                override = "⚠ CYCLONE LANDFALL"
            elif day_offset == 4:
                rain_boost = 30.0 + random.uniform(-5, 5)
                precip_pred_mm = max(rain_boost, precip_pred_mm * 1.5)
                temp_pred += 1.0
                override = "⚠ Cyclone Remnants"
            else:
                precip_pred_mm = random.randrange(1,5) +    random.random()
                
            return temp_pred, precip_pred_mm, override

        latest = history_df.iloc[-1]
        pressure = latest.get('pres', 1010.0)
        wind = latest.get('wspd', 10.0)
        
        prev_pres = history_df.iloc[-2].get('pres', 1010.0) if len(history_df) > 1 else pressure
        pres_drop = prev_pres - pressure

        month = prediction_date.month
        is_season = month in [10, 11, 12, 4, 5]
        
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