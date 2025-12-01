# Weather Prediction App - Setup & Run

Prerequisites
- Python 3.12.10 (please install this exact version before proceeding)

Setup
1. Clone the repository and open a terminal in its root directory.
2. Run the provided requirements script which will create a virtual environment and install all required libraries:

   On macOS / Linux:
   python requirements.py

   On Windows (with the py launcher):
   py requirements.py

3. Activate the virtual environment if the script created one (common locations):

   macOS / Linux:
   source venv/bin/activate

   Windows (PowerShell):
   .\venv\Scripts\Activate.ps1

   Windows (cmd):
   .\venv\Scripts\activate

Running the app (order matters)
1. First, run the script that downloads or prepares historical weather data. The repository contains the script for this step; its filename may be "get historical data.py" (or a similarly named file). Example:

   python "get_historical_data.py"

   or on Windows:
   py "get_historical_data.py"

   If you don't see a file with that exact name, look for a script containing "histor", "data", or "get" in its filename and run it first.

2. Next, run the data-preprocessing / feature extraction scripts in this order:

   python precipitator_predictor (or) python3 main.py
   python Linear_reg.py (or) python3 main.py


   These scripts train or prepare models using the historical data. Depending on your machine and dataset size, training may take time.

3. After training completes, run the main application which shows predicted values:

   python main.py (or) python3 main.py


Notes & troubleshooting
- If the repository's filenames differ (for example underscores or hyphens), run the script that clearly fetches historical data first, then the two training scripts, then main.py.
- Ensure your internet connection is available if the historical-data script downloads data from external sources.
- If you encounter missing package errors, make sure the virtual environment was activated and that requirements.py completed successfully. You can also manually install dependencies with pip inside the venv.
- If you want, I can update this README to include exact filenames after I inspect the repository — say the word "inspect" and I'll look up the actual script names and update the README accordingly.
