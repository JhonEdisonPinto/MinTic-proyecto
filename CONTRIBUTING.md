## Contribuir al proyecto MinTIC

Gracias por contribuir. Este documento describe el flujo de trabajo recomendado.

### Flujo básico (Git):
- Fork del repositorio (si no tienes permiso directo)
- Crear una rama de feature: `git checkout -b feat/mi-cambio`
- Hacer commits pequeños y descriptivos
- Push a tu fork/branch y abrir Pull Request

### Reglas y herramientas:
- Formatear con `black` antes de push: `black .`
- Ejecutar linters: `flake8 src tests`
- Ejecutar tests: `pytest`
- Instalar hooks: `pre-commit install` (recomendado)

### Revisión de PR:
- Añade descripción clara y pasos para reproducir
- Vincula issues relacionados
- Solicita revisión a un miembro del equipo

### Etiqueta de ramas:
- `feat/...`, `fix/...`, `chore/...`, `docs/...`, `test/...`
