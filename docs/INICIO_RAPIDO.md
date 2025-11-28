# 🚗 GUÍA DE INICIO RÁPIDO - STREAMLIT

## ⚡ 3 pasos para iniciar la aplicación

### Paso 1: Abrir PowerShell

En Windows, presiona:
- Win + X → Windows PowerShell o Terminal

O navega a la carpeta del proyecto.

### Paso 2: Ejecutar el script

```powershell
.\RUN_STREAMLIT.ps1
```

Si tienes permisos de ejecución restringidos:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\RUN_STREAMLIT.ps1
```

### Paso 3: Usar la aplicación

La app se abrirá automáticamente en `http://localhost:8501`

Si no se abre, copia y pega la URL en tu navegador.

---

## 🛠️ Si hay problemas

### Problema: "Streamlit no está instalado"

```powershell
.\INSTALL_STREAMLIT.ps1
.\RUN_STREAMLIT.ps1
```

### Problema: "ModuleNotFoundError"

Asegúrate de que estés en la carpeta correcta:

```powershell
cd C:\Users\Jhon\Documents\GitHub\MinTic-proyecto
.\RUN_STREAMLIT.ps1
```

### Problema: "GEMINI_API_KEY no configurada"

Edita el archivo `.env`:

```env
GEMINI_API_KEY=tu-clave-aqui
```

---

## 📊 Qué puedes hacer en la aplicación

### 1. 📄 Análisis de PDF
- Lee la Ley 769 de 2002 automáticamente
- Haz preguntas sobre documentos legales
- Ve el texto extraído

### 2. 📈 Exploración de Datos
- Analiza siniestros viales en Palmira (2,834 casos)
- Ve gráficos y estadísticas
- Haz preguntas sobre los datos

### 3. 🔗 Análisis Unificado
- Combina PDF + datos
- Haz preguntas que mezclen ley con estadísticas

### 4. 📋 Reportes
- Ve gráficos automáticos
- Descarga datos

---

## 💡 Ejemplos de preguntas

**Sobre PDF:**
- "¿Qué sanciones tiene conducir embriagado?"
- "¿Qué dice la ley sobre CHOQUES?"

**Sobre datos:**
- "¿Cuál es el tipo de siniestro más frecuente?"
- "¿En qué hora ocurren más accidentes?"

**Combinadas:**
- "¿El CHOQUE es frecuente y qué dice la ley?"

---

## 🎯 Archivos importantes

```
✅ app/streamlit_app.py ......... Aplicación (este es el archivo principal)
✅ .env ......................... Configuración (necesita GEMINI_API_KEY)
✅ data/Ley_769_de_2002.pdf ..... Documento legal
✅ data/siniestros_1_limpio.csv . Datos de siniestros viales en Palmira
```

---

## 📞 Si algo no funciona

1. Abre PowerShell en la carpeta del proyecto
2. Escribe: `.\RUN_STREAMLIT.ps1`
3. Espera a que aparecer el mensaje "Iniciando aplicación"
4. La app se abrirá automáticamente

Si aún hay problemas:
- Verifica que tienes `.env` con `GEMINI_API_KEY`
- Comprueba que tienes los archivos en `data/`
- Reinicia PowerShell

---

## 🎓 Componentes técnicos

La app integra:
- ✅ OCR (extrae PDF automáticamente)
- ✅ Análisis CSV (2,834 registros)
- ✅ Gemini API (respuestas inteligentes)
- ✅ Gráficos interactivos
- ✅ Reportes automáticos

---

**¡Listo! Ya puedes usar la aplicación de análisis de siniestros viales.**

---

Fecha: Noviembre 2025
