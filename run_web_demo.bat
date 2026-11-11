@echo off
title ASVS Banking Signature Verification Demo
cd /d "%~dp0"

echo ======================================================================
echo    HE THONG XAC THUC CHU KY NGAN HANG TU DONG (ASVS DEMO)
echo    Module 05: Deep Learning Reusing Pretrained Models
echo ======================================================================
echo.
echo [*] Dang khoi dong mo hinh ResNet-50 va may chu web...
echo [*] Trinh duyet web se tu dong bat len tai: http://127.0.0.1:5000
echo.

"C:\Users\ADMIN\AppData\Local\Programs\Python\Python312\python.exe" app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Co loi xay ra khi khoi chay.
    pause
)
