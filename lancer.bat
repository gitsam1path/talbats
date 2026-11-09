@echo off
chcp 65001 >nul
title TalBats
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Installation de l'environnement Python ^(une seule fois^)...
  python -m venv .venv || (echo Python introuvable : installez Python 3.10+ depuis python.org & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
echo Lancement de l'application sur http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
