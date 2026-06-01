@echo off
setlocal
cd /d %~dp0
python -m src.train_model
python build_presentation.py
echo.
echo Project training and presentation generation completed.
pause
