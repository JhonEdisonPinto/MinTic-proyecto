# Script para ejecutar la aplicación Streamlit
# Uso: .\RUN_STREAMLIT.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  🚗 INICIALIZANDO APLICACIÓN STREAMLIT" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Cambiar al directorio del proyecto
$projectPath = "C:\Users\Jhon\Documents\GitHub\MinTic-proyecto"
Set-Location $projectPath

# Verificar que venv_nuevo existe
if (-not (Test-Path "venv_nuevo")) {
    Write-Host "❌ Entorno virtual no encontrado en venv_nuevo" -ForegroundColor Red
    Write-Host "Por favor, crea el entorno virtual primero." -ForegroundColor Yellow
    exit 1
}

# Activar entorno virtual
Write-Host "📦 Activando entorno virtual..." -ForegroundColor Yellow
& ".\venv_nuevo\Scripts\Activate.ps1"

# Verificar que Streamlit está instalado (uso seguro de comillas)
Write-Host "🔍 Verificando dependencias..." -ForegroundColor Yellow
# Intentar obtener la versión de streamlit de forma segura
$streamlitVersion = $null
try {
    $streamlitVersion = & python -c 'import streamlit; print(streamlit.__version__)' 2>$null
} catch {
    $streamlitVersion = $null
}
if ([string]::IsNullOrEmpty($streamlitVersion)) {
    Write-Host "⚠️  Streamlit no está instalado. Instalando..." -ForegroundColor Yellow
    pip install streamlit --quiet
    # Intentar nuevamente obtener la versión
    try { $streamlitVersion = & python -c 'import streamlit; print(streamlit.__version__)' 2>$null } catch { $streamlitVersion = $null }
}

# Mostrar información
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ PREPARADO PARA INICIAR" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 Ubicación: $projectPath" -ForegroundColor White
Write-Host "🐍 Python: $(python --version)" -ForegroundColor White
if ($streamlitVersion) {
    Write-Host "📊 Streamlit: $streamlitVersion" -ForegroundColor White
} else {
    Write-Host "📊 Streamlit: (no disponible)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "🚀 Iniciando aplicación en http://localhost:8501" -ForegroundColor Cyan
Write-Host "   (Presiona Ctrl+C para detener)" -ForegroundColor Gray
Write-Host ""

# Limpiar cache de Python
Write-Host "Limpiando cache..." -ForegroundColor Yellow
Get-ChildItem -Path src,app -Recurse -Filter "__pycache__" -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path src,app -Recurse -Filter "*.pyc" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "Cache limpiado correctamente" -ForegroundColor Green
Write-Host ""

# Ejecutar Streamlit
streamlit run app/streamlit_app.py
