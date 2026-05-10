@echo off
echo Creating virtual environment...
python -m venv venv
echo Activating virtual environment...
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo Generating synthetic data...
python src/generate_synthetic_data.py
echo Training models...
python src/train_baseline.py
python src/train_xgboost.py
echo Setup complete!
echo.
echo To start the system:
echo 1. Start API: cd api && python app.py
echo 2. Start Dashboard: streamlit run dashboard/app_streamlit.py
pause
