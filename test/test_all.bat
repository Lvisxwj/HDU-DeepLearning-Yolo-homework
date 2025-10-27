@echo off
echo ========================================
echo Testing All Images
echo ========================================
echo.

call conda activate yolov11
python test_all_images.py

pause
