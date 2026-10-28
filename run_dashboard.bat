@echo off
TITLE SPAMIX - SMS Spam Detection AI Dashboard
echo ======================================================================
echo          SPAMIX - SMS SPAM DETECTION USING DEEP LEARNING
echo                     B.Tech Final Year Project
echo ======================================================================
echo.
echo Launching SPAMIX Interactive Analytics Dashboard...
cd /d "%~dp0"
streamlit run dashboard/app.py
pause
