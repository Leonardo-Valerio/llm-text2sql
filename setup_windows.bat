@echo off
REM Uso: setup_windows.bat [cu126|cu128|cu124|cu118]   (padrao: cu126)
set CU=%1
if "%CU%"=="" set CU=cu126
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/%CU%
pip install -r requirements.txt
python check_env.py
