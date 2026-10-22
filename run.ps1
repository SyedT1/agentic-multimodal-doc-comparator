$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    throw 'Create the virtual environment first: py -3.11 -m venv .venv'
}
& $appPython -X utf8 -m streamlit run src/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
