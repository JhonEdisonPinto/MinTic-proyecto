# Script para instalar dependencias de Streamlit
# Uso: .\INSTALL_STREAMLIT.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  📦 INSTALANDO STREAMLIT" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Cambiar al directorio del proyecto
$projectPath = "C:\Users\Jhon\Documents\GitHub\MinTic-proyecto"
Set-Location $projectPath

# Verificar que .venv existe
if (-not (Test-Path ".venv")) {
    Write-Host "❌ Entorno virtual no encontrado en .venv" -ForegroundColor Red
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activar entorno virtual
Write-Host "📦 Activando entorno virtual..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Actualizar pip
Write-Host "🔄 Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Instalar Streamlit
Write-Host "📥 Instalando Streamlit..." -ForegroundColor Yellow
pip install streamlit

# Mostrar versión instalada
$version = python -c "import streamlit; print(streamlit.__version__)"
Write-Host "✅ Streamlit $version instalado correctamente" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✅ INSTALACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para ejecutar la aplicación, usa:" -ForegroundColor Yellow
Write-Host "  .\RUN_STREAMLIT.ps1" -ForegroundColor White
Write-Host ""
