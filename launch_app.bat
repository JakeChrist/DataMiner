@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

if exist "%SCRIPT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe"
) else (
    call :find_python
    if not defined PYTHON (
        echo Could not locate Python. Ensure Python 3.11+ is installed or create a .venv folder.
        exit /b 1
    )
)

pushd "%SCRIPT_DIR%"
"%PYTHON%" -m app %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %APP_EXIT_CODE%

:find_python
for %%P in (python.exe py.exe) do (
    where %%P >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=%%P"
        goto :eof
    )
)
goto :eof

