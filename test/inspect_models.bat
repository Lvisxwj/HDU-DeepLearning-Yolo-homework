@echo off
echo ========================================
echo Inspecting Custom Model Classes
echo ========================================
echo.

call conda activate yolov11
python inspect_models.py

pause
