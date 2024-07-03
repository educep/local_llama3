@echo off
SET PROJECT_NAME=local_llama3
SET PYTHON_INTERPRETER=python
IF "%~1" NEQ "" SET PROJECT_NAME=%~1
IF "%~2" NEQ "" SET PYTHON_INTERPRETER=%~2

IF EXIST %PROJECT_NAME% (
    echo Virtual environment %PROJECT_NAME% already exists.
) ELSE (
    %PYTHON_INTERPRETER% -m venv %PROJECT_NAME%
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment.
        exit /b 1
    ) ELSE (
        echo Virtual environment %PROJECT_NAME% created successfully.
    )
)
