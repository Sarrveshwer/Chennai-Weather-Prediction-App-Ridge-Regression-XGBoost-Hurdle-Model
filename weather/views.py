from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from datetime import timedelta
import contextlib
import io
import joblib
import os
import sys

# Add parent directory to path to import weather_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the WeatherForecastingEngine class so joblib can unpickle it
from weather_engine import WeatherForecastingEngine

# Global variable to cache the engine
_engine_cache = None

def get_engine():
    """Load and cache the weather engine model"""
    global _engine_cache
    
    if _engine_cache is not None:
        return _engine_cache
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models', 'weather_engine.joblib')
    
    if os.path.exists(model_path):
        try:
            # Workaround for joblib unpickling: add class to __main__ module
            import __main__
            __main__.WeatherForecastingEngine = WeatherForecastingEngine
            
            _engine_cache = joblib.load(model_path)
            print(f"Weather engine loaded successfully from {model_path}")
            return _engine_cache
        except Exception as e:
            print(f"Error loading weather engine: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"Weather engine model not found at {model_path}")
        return None


def index(request):
    engine = get_engine()
    context = {
        'current_temp': None,
        'current_precip': None,
        'status': 'offline'
    }
    
    if engine:
        try:
            current_data = engine.get_data_slice(days=50)
            last = current_data.iloc[-1]
            context['current_temp'] = last['tavg']
            context['current_precip'] = last['prcp']
            context['status'] = 'online'
        except Exception as e:
            print(f"Error loading current data: {e}")
    
    return render(request, 'weather/index.html', context)

def generate_forecast(request):
    engine = get_engine()
    if not engine:
        return JsonResponse({'error': 'Engine Offline'}, status=500)
    
    try:
        days = int(request.GET.get('days', 7))
        days = max(1, min(days, 14))
        
        # Support test_date parameter for testing (format: YYYY-MM-DD)
        test_date = request.GET.get('test_date', None)
        
        engine.fetch_data()
        current_data = engine.get_data_slice(days=100)
        history = current_data.copy()
        
        if test_date:
            try:
                # Override the current date for testing
                current_cursor = pd.to_datetime(test_date)
                print(f"Using test date: {current_cursor}")
            except:
                # Fall back to actual current date if test_date is invalid
                current_cursor = pd.to_datetime(history.iloc[-1]['date'])
        else:
            current_cursor = pd.to_datetime(history.iloc[-1]['date'])
        
        forecasts = []
        
        for i in range(1, days + 1):
            target_date = current_cursor + timedelta(days=i)
            
            new_row = history.iloc[-1].copy()
            new_row['date'] = target_date
            new_row['tavg'], new_row['prcp'] = np.nan, np.nan
            
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            
            with contextlib.redirect_stdout(io.StringIO()):
                df_feats = engine.engineer_features(history)
            
            X_vector = df_feats.iloc[[-1]][engine.feature_cols]
            X_scaled = engine.scaler.transform(X_vector)
            
            t_pred = float(engine.temp_model.predict(X_vector)[0])
            prob = float(engine.rain_classifier.predict_proba(X_scaled)[0][1])
            
            p_pred = 0.0
            if prob > 0.45:
                raw_preds = [m.predict(X_scaled)[0] for m in engine.rain_regressors]
                p_pred = np.mean([np.expm1(p) for p in raw_preds])
            
            history.iloc[-1, history.columns.get_loc('tavg')] = t_pred
            history.iloc[-1, history.columns.get_loc('prcp')] = p_pred
            
            forecasts.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'day': target_date.strftime('%a'),
                'full_date': target_date.strftime('%B %d, %Y'),
                'temp': t_pred,
                'precip': p_pred,
                'prob': int(prob * 100),
                'icon': 'rain' if prob > 0.5 else 'sun'
            })
            
        return JsonResponse({'forecasts': forecasts})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_alerts(request):
    """Generate weather alerts for Chennai based on current conditions from Open-Meteo"""
    try:
        import requests
        from datetime import datetime, timedelta
        
        # Use Open-Meteo API (same as we use for forecasts)
        # Get current weather conditions for Chennai
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': 13.0836939,
            'longitude': 80.270186,
            'current': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code',
            'timezone': 'Asia/Kolkata'
        }
        
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            alerts = []
            
            if 'current' in data:
                current = data['current']
                
                temp = current.get('temperature_2m', 0)
                humidity = current.get('relative_humidity_2m', 0)
                wind_speed = current.get('wind_speed_10m', 0)
                precip = current.get('precipitation', 0)
                weather_code = current.get('weather_code', 0)
                
                # Generate alerts based on extreme conditions
                # Heat wave (temperature > 38°C)
                if temp > 38:
                    alerts.append({
                        'title': 'Heat Wave Warning',
                        'description': f'High temperature of {temp:.1f}°C. Stay hydrated, avoid direct sunlight during peak hours (11 AM - 4 PM), and check on elderly neighbors.',
                        'severity': 'orange',
                        'valid_until': (datetime.now() + timedelta(hours=12)).strftime('%Y-%m-%d %H:%M')
                    })
                
                # Very hot (temperature > 35°C)
                elif temp > 35:
                    alerts.append({
                        'title': 'Hot Weather Advisory',
                        'description': f'Temperature at {temp:.1f}°C. Drink plenty of water and limit outdoor activities.',
                        'severity': 'yellow',
                        'valid_until': (datetime.now() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
                    })
                
                # Strong wind (> 40 km/h)
                if wind_speed > 40:
                    alerts.append({
                        'title': 'Strong Wind Advisory',
                        'description': f'Wind speeds up to {wind_speed:.1f} km/h. Secure loose objects and avoid coastal areas.',
                        'severity': 'yellow',
                        'valid_until': (datetime.now() + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')
                    })
                
                # Heavy rain (weather code 95-99 are thunderstorms)
                if weather_code >= 95:
                    alerts.append({
                        'title': 'Thunderstorm Warning',
                        'description': 'Thunderstorms expected. Stay indoors, avoid using electrical appliances, and do not take shelter under trees.',
                        'severity': 'orange',
                        'valid_until': (datetime.now() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M')
                    })
                
                # Moderate to heavy rain (weather code 61-67, 80-82)
                elif weather_code in [61, 63, 65, 67, 80, 81, 82]:
                    alerts.append({
                        'title': 'Heavy Rain Alert',
                        'description': 'Heavy rainfall expected. Carry an umbrella and avoid waterlogged areas.',
                        'severity': 'yellow',
                        'valid_until': (datetime.now() + timedelta(hours=6)).strftime('%Y-%m-%d %H:%M')
                    })
                
                # High humidity (> 90%)
                if humidity > 90 and temp > 28:
                    alerts.append({
                        'title': 'High Humidity Alert',
                        'description': f'Humidity at {humidity}% with temperature {temp:.1f}°C. Expect muggy and uncomfortable conditions.',
                        'severity': 'green',
                        'valid_until': (datetime.now() + timedelta(hours=12)).strftime('%Y-%m-%d %H:%M')
                    })
            
            return JsonResponse({'alerts': alerts})
        
        else:
            # Fallback: return empty alerts if API fails
            return JsonResponse({'alerts': []})
    
    except Exception as e:
        # Log error but don't crash - return empty alerts
        print(f"Error fetching weather alerts: {e}")
        return JsonResponse({'alerts': []})