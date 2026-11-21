@echo off
rem start.bat - start GLaDOS on Windows.   Double-click, or run:  start.bat
rem
rem   start.bat          LLM server + voice + listening + sim + GUI
rem   start.bat vision   camera + vision demo on this PC instead (serves :8081, like the Jetson)
rem
rem Vision normally comes from the Jetson (start it first with ./start.sh).
rem Options (set before running):
rem   set JETSON_IP=100.73.92.3    Jetson address; "none" = run without vision
rem   set VISION_MODE=auto         auto: Jetson if it answers, else this PC's webcam
rem                                (needs "setup.bat vision"), else none.  jetson / local / none
rem   set GLADOS_GAP_MS=480        silence before an utterance counts as finished
rem   set LLAMA_ARGS=-t 8          extra llama-server flags
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not defined WEB_PORT set "WEB_PORT=8081"

if /i "%~1"=="vision" goto vision

if not exist "listening\venv\Scripts\python.exe" (
    echo Not set up yet - run setup.bat first.
    pause
    exit /b 1
)
if not defined JETSON_IP set "JETSON_IP=100.73.92.3"
set "VISION_DEMO_PORT=%WEB_PORT%"

rem A previous run that did not exit cleanly would still hold the port and the model
taskkill /F /IM llama-server.exe >nul 2>&1

echo GLaDOS - Windows   (vision from: %JETSON_IP%)
"listening\venv\Scripts\python.exe" gui_launch.py
goto :eof

:vision
if not exist "vision\demo\.venv\Scripts\python.exe" (
    echo Vision is not set up yet - run:  setup.bat vision
    pause
    exit /b 1
)
echo GLaDOS vision - http://localhost:%WEB_PORT%/   (in the other window: set JETSON_IP=127.0.0.1)
"vision\demo\.venv\Scripts\python.exe" vision\demo\run_demo.py --source auto --no-window --web %WEB_PORT%
