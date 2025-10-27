@echo off
echo ========================================
echo Testing Custom Models
echo ========================================
echo.

REM Activate conda environment
call conda activate yolov11

REM Run custom model test
python test_custom_models.py

pause
