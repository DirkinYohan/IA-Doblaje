@echo off
REM ============================================================
REM  IA-Doblaje - arranca el motor (API) y la app web a la vez.
REM  Doble clic sobre este archivo. Se abren dos ventanas.
REM ============================================================
setlocal

set "REPO=%~dp0"
set "WEB=%REPO%..\video_subtitles"
set "PY=%REPO%.venv\Scripts\python.exe"

echo Comprobando el entorno...

if not exist "%PY%" (
  echo.
  echo  ERROR: no encuentro el entorno de Python en:
  echo    %PY%
  echo.
  echo  Creelo con:
  echo    python -m venv .venv
  echo    .venv\Scripts\python.exe -m pip install -e ".[dev,asr,diarization,vad]"
  echo.
  pause
  exit /b 1
)

if not exist "%WEB%\package.json" (
  echo.
  echo  ERROR: no encuentro el frontend en:
  echo    %WEB%
  echo.
  pause
  exit /b 1
)

echo.
echo  1) Motor + API  -^>  http://127.0.0.1:8000/docs
echo  2) App web      -^>  http://127.0.0.1:3000
echo.
echo  Deja las DOS ventanas abiertas mientras uses la aplicacion.
echo  Para parar todo: cierra las dos ventanas, o pulsa Ctrl+C en cada una.
echo.

REM --- Ventana 1: motor de IA + API (debe arrancar desde la raiz del repo) ---
start "IA-Doblaje API  (NO CERRAR)" cmd /k "cd /d "%REPO%" && "%PY%" -m app.presentation.api"

REM --- Ventana 2: frontend Next.js ---
start "IA-Doblaje Web  (NO CERRAR)" cmd /k "cd /d "%WEB%" && npm run dev"

echo Esperando a que arranquen...
timeout /t 12 /nobreak >nul

echo.
echo  Abriendo el navegador...
start "" "http://127.0.0.1:3000"

echo.
echo  Si el navegador muestra un error, espera 10 segundos y recarga
echo  (la primera compilacion de Next.js tarda un poco).
echo.
pause
