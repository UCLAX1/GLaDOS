@echo off
rem setup.bat - one-time setup on Windows.   Double-click, or run:  setup.bat
rem
rem   setup.bat          brain + voice + listening + sim + GUI   (listening\venv)
rem   setup.bat vision   also set up the camera/vision demo on this PC (vision\demo\.venv)
rem
rem Safe to re-run; it only installs what is missing.
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo.
echo === GLaDOS setup - Windows ===

rem --- Python 3.10-3.12 (MediaPipe / PyAudio have no wheels for newer versions yet)
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo Python 3.10-3.12 not found. Install Python 3.12 from python.org and re-run.
    pause
    exit /b 1
)
echo   [ok] Python: %PY%

rem --- Virtual environment
if not exist "listening\venv\Scripts\python.exe" (
    echo   creating listening\venv ...
    %PY% -m venv listening\venv
)
set "VPY=listening\venv\Scripts\python.exe"
"%VPY%" -m pip install -q --upgrade pip

rem --- NVIDIA GPU? Then install the CUDA build of PyTorch before everything else.
set "GPU=cpu"
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "GPU=nvidia"
echo   [ok] GPU: %GPU%
if "%GPU%"=="nvidia" (
    echo   installing PyTorch with CUDA ...
    "%VPY%" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu124
)
echo   installing Python packages - first run takes a few minutes ...
"%VPY%" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo   [x] pip install failed - see the messages above.
    pause
    exit /b 1
)
echo   [ok] Python packages

rem --- llama-server (PrismML build of llama.cpp, which runs the Bonsai model)
where llama-server >nul 2>&1
if not errorlevel 1 (
    echo   [ok] llama-server ^(on PATH^)
) else (
    dir /s /b "tools\llama\llama-server.exe" >nul 2>&1
    if not errorlevel 1 (
        echo   [ok] llama-server ^(tools\llama^)
    ) else (
        set "TAG=prism-b12128-d7b2896"
        set "BASE=https://github.com/PrismML-Eng/llama.cpp/releases/download/!TAG!"
        if "%GPU%"=="nvidia" (
            set "ZIPS=llama-!TAG!-bin-win-cuda-12.4-x64.zip cudart-llama-bin-win-cuda-12.4-x64.zip"
        ) else (
            set "ZIPS=llama-!TAG!-bin-win-cpu-x64.zip"
        )
        if not exist "tools\llama" mkdir "tools\llama"
        for %%Z in (!ZIPS!) do (
            echo   downloading %%Z ...
            powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '!BASE!/%%Z' -OutFile 'tools\llama\%%Z'; Expand-Archive -Force 'tools\llama\%%Z' 'tools\llama'; Remove-Item 'tools\llama\%%Z'"
        )
        dir /s /b "tools\llama\llama-server.exe" >nul 2>&1
        if errorlevel 1 (
            echo   [x] could not download llama-server. Get a Windows build from
            echo        https://github.com/PrismML-Eng/llama.cpp/releases and unzip it into tools\llama
        ) else (
            echo   [ok] llama-server ^(tools\llama^)
        )
    )
)

rem --- Model
set "GGUF=central_ai\benchmarking_tools\Bonsai-8B-Q1_0.gguf"
if exist "%GGUF%" (
    echo   [ok] model
) else (
    echo   downloading Bonsai-8B ^(1.1 GB^) ...
    "%VPY%" -m pip install -q huggingface_hub
    "%VPY%" -c "from huggingface_hub import hf_hub_download as d; d('prism-ml/Bonsai-8B-gguf','Bonsai-8B-Q1_0.gguf',local_dir='central_ai/benchmarking_tools')"
    if not exist "%GGUF%" echo   [x] download failed - put Bonsai-8B-Q1_0.gguf into central_ai\benchmarking_tools
)

rem --- Optional: vision demo on this PC
if /i "%~1"=="vision" (
    echo.
    echo === Vision demo ===
    %PY% vision\demo\setup_demo.py
)

echo.
echo Done. Start with:  start.bat
pause
