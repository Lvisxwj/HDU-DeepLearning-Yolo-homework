@echo off
echo ========================================
echo Vehicle Submersion Detection System
echo ========================================
echo.
echo Starting application...
echo.

REM Activate conda environment
call conda activate yolov11

REM Run Streamlit app
streamlit run app.py

pause
