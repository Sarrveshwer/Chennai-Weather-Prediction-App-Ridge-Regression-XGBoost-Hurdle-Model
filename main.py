#!/usr/bin/env python3
"""
Simple launcher for Weather Prediction Django app.
Checks venv, installs dependencies, trains model if needed, then runs server.
"""

import os
import sys
import subprocess
from pathlib import Path

# Setup logging
from auto_logger import setup_logging
setup_logging("main_launcher")

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"[✓] {description} - Done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[✗] {description} - Failed")
        return False

def main():
    print("\n" + "="*50)
    print("Weather Prediction App Launcher")
    print("="*50)
    
    # Check if .venv exists
    if not Path('.venv').exists():
        print("\n[!] Virtual environment not found")
        print("[*] Creating .venv...")
        if not run_command(f"{sys.executable} -m venv .venv", "Create virtual environment"):
            sys.exit(1)
        
        print("\n[✓] Virtual environment created")
        print("[*] Installing dependencies...")
        
        # Determine pip path based on OS
        if os.name == 'nt':  # Windows
            pip_path = ".venv\\Scripts\\pip"
        else:  # Linux/Mac
            pip_path = ".venv/bin/pip"
        
        # Install dependencies using the venv's pip
        if Path('requirements.txt').exists():
            run_command(f"{pip_path} install -r requirements.txt", "Install dependencies")
        
        print("\n[!] Setup complete! Please activate the virtual environment and run this script again:")
        if os.name == 'nt':  # Windows
            print("    .venv\\Scripts\\activate")
        else:  # Linux/Mac
            print("    source .venv/bin/activate")
        print("    python main.py")
        sys.exit(0)
    
    # Check if we're in venv
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("\n[!] Virtual environment exists but not activated")
        print("[!] Please activate it first:")
        if os.name == 'nt':
            print("    .venv\\Scripts\\activate")
        else:
            print("    source .venv/bin/activate")
        print("    python main.py")
        sys.exit(0)
    
    print("\n[✓] Virtual environment is active")
    
    
    # Check for model file
    model_path = Path('saved_models/weather_engine.joblib')
    if not model_path.exists():
        print("\n[!] Model file not found")
        print("[*] Training model (this will take ~30 seconds)...")
        if not run_command(f"{sys.executable} weather_engine.py", "Train weather model"):
            print("\n[✗] Model training failed")
            sys.exit(1)
    else:
        print(f"\n[✓] Model file found ({model_path.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Run Django server
    print("\n" + "="*50)
    print("Starting Django server...")
    print("="*50)
    print("\nServer will be available at: http://127.0.0.1:8000/")
    print("Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, 'manage.py', 'runserver'])
    except KeyboardInterrupt:
        print("\n\n[*] Server stopped")

if __name__ == "__main__":
    main()
