@echo off
title Lead Scoring Pipeline
color 0A

chcp 65001 >nul
set PYTHONENCODING=utf-8
:: Python'un PATH'te olduğunu kontrol et
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python bulunamadi! Lutfen Python'u yukleyin ve PATH'e ekleyin.
    pause
    exit /b 1
)

:: Gerekli paketleri yüklediğimizden emin ol
echo Gerekli paketler kontrol ediliyor...
python -c "import pandas, numpy, sklearn, matplotlib, streamlit, mlflow, click" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Bazi gerekli paketler eksik. Kurulum baslatiliyor...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Paket kurulumu basarisiz oldu. Lutfen manuel olarak 'pip install -r requirements.txt' komutunu calistirin.
        pause
        goto MENU
    )
    echo Paketler basariyla kuruldu!
)

:: Çalışmadan önce temizlik dosyasını çalıştır
python -c "from src.utils.cleanup import cleanup_old_dirs; cleanup_old_dirs()"

:MENU
cls
echo ===============================
echo    LEAD SCORING PIPELINE
echo ===============================
echo.
echo  1. Tam Pipeline Calistir
echo  2. Istatistiksel Analizler
echo  3. Feature Importance Hesapla
echo  4. Model Egit
echo  5. Dashboard Baslat
echo  6. MLflow UI Baslat
echo  7. Tum Servisleri Baslat
echo  8. Cikis
echo.
echo ===============================
echo.

set /p choice=Secim yapiniz (1-8): 

if "%choice%"=="1" goto FULLRUN
if "%choice%"=="2" goto STATS
if "%choice%"=="3" goto FEATIMP
if "%choice%"=="4" goto TRAIN
if "%choice%"=="5" goto DASHBOARD
if "%choice%"=="6" goto MLFLOW
if "%choice%"=="7" goto ALLSERVICES
if "%choice%"=="8" goto END

echo Gecersiz secim. Lutfen tekrar deneyin.
timeout /t 2 >nul
goto MENU

:FULLRUN
cls
echo Tam pipeline calistiriliyor...
python lead_scoring.py all
echo.
echo Islem tamamlandi.
pause
goto MENU

:STATS
cls
echo Istatistiksel analizler yapiliyor...
echo.
echo Hangi testi calistirmak istiyorsunuz?
echo  1. Ki-kare Testi
echo  2. T-Test
echo  3. ANOVA
echo  4. Korelasyon Analizi
echo  5. Donusum Orani Karsilastirmasi
echo  6. Tum Testler
echo  7. Geri
echo.
set /p test_choice=Secim yapiniz (1-7): 

if "%test_choice%"=="1" (
    python lead_scoring.py stats --test_type=chi_square
) else if "%test_choice%"=="2" (
    python lead_scoring.py stats --test_type=t_test --group_col=Source_Final__c
) else if "%test_choice%"=="3" (
    python lead_scoring.py stats --test_type=anova --group_col=Source_Final__c
) else if "%test_choice%"=="4" (
    python lead_scoring.py stats --test_type=correlation
) else if "%test_choice%"=="5" (
    python lead_scoring.py stats --test_type=conversion_rate
) else if "%test_choice%"=="6" (
    python lead_scoring.py stats --test_type=all
) else if "%test_choice%"=="7" (
    goto MENU
) else (
    echo Gecersiz secim.
)

echo.
echo Islem tamamlandi.
pause
goto MENU

:FEATIMP
cls
echo Feature importance hesaplaniyor...
echo.
echo Hangi yontemi kullanmak istiyorsunuz?
echo  1. SHAP
echo  2. Permutation
echo  3. Her Ikisi
echo  4. Geri
echo.
set /p method_choice=Secim yapiniz (1-4): 

if "%method_choice%"=="1" (
    python lead_scoring.py feature-imp --method=shap
) else if "%method_choice%"=="2" (
    python lead_scoring.py feature-imp --method=permutation
) else if "%method_choice%"=="3" (
    python lead_scoring.py feature-imp --method=both
) else if "%method_choice%"=="4" (
    goto MENU
) else (
    echo Gecersiz secim.
)

echo.
echo Islem tamamlandi.
pause
goto MENU

:TRAIN
cls
echo Model egitiliyor...
echo.
echo Hangi model tipini egitmek istiyorsunuz?
echo  1. Baseline (LogisticRegression)
echo  2. LightGBM
echo  3. Source-based
echo  4. Ensemble
echo  5. Geri
echo.
set /p model_choice=Secim yapiniz (1-5): 

if "%model_choice%"=="1" (
    python lead_scoring.py train --model-type=1
) else if "%model_choice%"=="2" (
    python lead_scoring.py train --model-type=2
) else if "%model_choice%"=="3" (
    python lead_scoring.py train --model-type=3
) else if "%model_choice%"=="4" (
    python lead_scoring.py train --model-type=4
) else if "%model_choice%"=="5" (
    goto MENU
) else (
    echo Gecersiz secim.
)

echo.
echo Islem tamamlandi.
pause
goto MENU

:DASHBOARD
cls
echo Streamlit dashboard baslatiliyor...
start /b "" cmd /c python lead_scoring.py dashboard --no-browser
echo.
echo Dashboard baslatildi. Tarayicinizda http://localhost:8501 adresine giderek goruntuleyebilirsiniz.
start http://localhost:8501
echo Ana menuye donmek icin herhangi bir tusa basin.
pause >nul
goto MENU

:MLFLOW
cls
echo MLflow UI baslatiliyor...
start /b "" cmd /c python lead_scoring.py mlflow --no-browser
echo.
echo MLflow UI baslatildi. Tarayicinizda http://localhost:5000 adresine giderek goruntuleyebilirsiniz.
start http://localhost:5000
echo Ana menuye donmek icin herhangi bir tusa basin.
pause >nul
goto MENU

:ALLSERVICES
cls
echo Tum servisler baslatiliyor...
start /b "" cmd /c python lead_scoring.py dashboard --no-browser
start /b "" cmd /c python lead_scoring.py mlflow --no-browser
echo.
echo Tum servisler baslatildi:
echo - Streamlit Dashboard: http://localhost:8501
echo - MLflow UI: http://localhost:5000
echo.
start http://localhost:8501
start http://localhost:5000
echo Ana menuye donmek icin herhangi bir tusa basin.
pause >nul
goto MENU

:END
echo Lead Scoring Pipeline kapatiliyor...
timeout /t 2 >nul
exit /b 0 