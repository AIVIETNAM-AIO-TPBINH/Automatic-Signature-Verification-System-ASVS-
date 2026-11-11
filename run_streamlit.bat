@echo off
chcp 65001 >nul
title [ASVS] Streamlit Banking Verification Demo
color 0A
cd /d "%~dp0"

echo ======================================================================
echo [*] KHOI DONG STREAMLIT DEMO: HE THONG XAC THUC CHU KY NGAN HANG
echo [*] Tac gia: Tran Phuong Binh (Module 05 - Deep Learning AIO2026)
echo [*] Dia chi truy cap: http://localhost:8501
echo [*] Nhan Ctrl+C de dong ung dung
echo ======================================================================
echo.

"C:\Users\ADMIN\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run streamlit_app.py --server.port 8501 --server.headless false

pause
