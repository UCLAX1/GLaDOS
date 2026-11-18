@echo off
rem start_demo.bat - start the vision demo on a Windows laptop (double-click, or run from a terminal).
rem Extra arguments go to run_demo.py, e.g.  start_demo.bat --fullscreen
rem First time: py -3.12 vision\demo\setup_demo.py   (needs internet)

set DEMO_DIR=%~dp0
if not exist "%DEMO_DIR%.venv\Scripts\python.exe" (
    echo Not set up yet. Run once:  py -3.12 "%DEMO_DIR%setup_demo.py"
    pause
    exit /b 1
)
cd /d "%DEMO_DIR%..\.."
"%DEMO_DIR%.venv\Scripts\python.exe" "%DEMO_DIR%run_demo.py" --source auto %*
pause
