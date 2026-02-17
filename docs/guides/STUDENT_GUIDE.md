# Guía para Estudiantes / Student Guide

> 🇦🇷 **Español abajo** — English first

---

## 🇬🇧 English Guide

### What is this?

BeamScan is our BL4S 2026 experiment simulation system. It lets you predict what will happen when we aim a particle beam at different materials. **You don't need to install anything** — GitHub runs everything for you in the cloud.

### How it works (3 minutes)

```
You edit a YAML file → push to GitHub → Actions runs automatically → plots appear
```

That's it. The YAML file is just a text file where you write what materials you want to simulate and at what beam energy. GitHub Actions reads your file, runs the Highland formula calculations, and produces plots + tables.

### Your first simulation (step by step)

**1. Go to the GitHub repository and click "Fork"** (top-right button).
This creates your own copy where you can experiment freely.

**2. Navigate to `requests/examples/`** and click on `template.yaml`.

**3. Click the pencil icon (✏️) to edit** the file in your browser.

**4. Click "Copy raw content"** and then navigate to `requests/` and create a new file.
Name it with your name: `requests/miNombre_miExperimento.yaml`

**5. Paste the template and edit these fields:**

```yaml
author: "Your Name"
description: "What you're investigating"

beam:
  particle: "e-"              # Don't change this for now
  momenta_GeV: [3.0, 6.0]    # Beam energies to simulate
  num_events: 10000           # How many particles

materials:
  - name: "PE"                            # Your label
    geant4_name: "G4_POLYETHYLENE"        # The physics name (pick from list)
    thickness_mm: 10.0                    # Target thickness

  - name: "PVC"
    geant4_name: "G4_POLYVINYL_CHLORIDE"
    thickness_mm: 10.0
```

**6. Commit** your changes (green button at bottom).

**7. Open a Pull Request** to the main repository.

**8. Wait ~1 minute** — GitHub Actions will run and post your results as a comment on the PR.

### What you'll get

- **📊 distributions.png** — Overlaid scattering curves for your materials (wider = more scattering)
- **📈 classification.png** — 2D plot showing how well materials separate
- **📋 predictions.csv** — Numbers: θ₀, energy loss, X₀ for each configuration
- **📝 SUMMARY.md** — Full report including how many events you need to tell each pair apart

### Ideas for investigations

| Difficulty | Investigation | What to change |
|-----------|--------------|---------------|
| ⭐ | Can we tell PVC from PE? | Simulate PE + PVC, check the distribution overlay |
| ⭐ | Does beam energy matter? | Same materials at [1.0, 3.0, 6.0, 10.0] GeV/c |
| ⭐⭐ | Thick vs thin targets | Same material at 5, 10, 20 mm thickness |
| ⭐⭐ | Heritage materials | Simulate obsidian, quartz, calcite, Fe₂O₃ |
| ⭐⭐⭐ | The PE vs PP challenge | Can ANY setting separate them? (Spoiler: very hard!) |
| ⭐⭐⭐ | Contaminant sensitivity | What if PVC has only 10% Cl? (custom material) |

### Troubleshooting

| Problem | Solution |
|---------|---------|
| Action didn't run | Make sure your file ends in `.yaml` and is in `requests/` (not `requests/examples/`) |
| YAML parse error | Check indentation (use spaces, not tabs). Use the [YAML validator](https://www.yamllint.com/) |
| Unknown material | Check the material list at the bottom of `template.yaml` |
| Plots look wrong | Check `thickness_mm` — did you mean 10.0 mm or 1.0 cm? |

---

## 🇦🇷 Guía en Español

### ¿Qué es esto?

BeamScan es el sistema de simulación de nuestro experimento BL4S 2026. Te permite predecir qué pasa cuando apuntamos un haz de partículas a diferentes materiales. **No necesitás instalar nada** — GitHub ejecuta todo por vos en la nube.

### Cómo funciona (3 minutos)

```
Editás un archivo YAML → lo subís a GitHub → Actions corre automáticamente → aparecen los gráficos
```

Eso es todo. El archivo YAML es un archivo de texto donde escribís qué materiales querés simular y a qué energía de haz. GitHub Actions lee tu archivo, ejecuta los cálculos de la fórmula de Highland, y produce gráficos + tablas.

### Tu primera simulación (paso a paso)

**1. Andá al repositorio en GitHub y hacé click en "Fork"** (botón arriba a la derecha).
Esto crea tu propia copia donde podés experimentar libremente.

**2. Navegá a `requests/examples/`** y hacé click en `template.yaml`.

**3. Hacé click en el ícono del lápiz (✏️)** para editar el archivo en tu navegador.

**4. Copiá el contenido** y creá un nuevo archivo en `requests/`.
Nombralo con tu nombre: `requests/miNombre_miExperimento.yaml`

**5. Pegá la plantilla y editá estos campos:**

```yaml
author: "Tu Nombre"
description: "Qué estás investigando"

beam:
  particle: "e-"              # No cambies esto por ahora
  momenta_GeV: [3.0, 6.0]    # Energías del haz a simular
  num_events: 10000           # Cuántas partículas

materials:
  - name: "PE"                            # Tu etiqueta
    geant4_name: "G4_POLYETHYLENE"        # Nombre en Geant4 (elegí de la lista)
    thickness_mm: 10.0                    # Espesor del blanco

  - name: "PVC"
    geant4_name: "G4_POLYVINYL_CHLORIDE"
    thickness_mm: 10.0
```

**6. Confirmá** tus cambios (botón verde abajo).

**7. Abrí un Pull Request** al repositorio principal.

**8. Esperá ~1 minuto** — GitHub Actions va a correr y publicar tus resultados como comentario en el PR.

### Qué vas a obtener

- **📊 distributions.png** — Curvas de dispersión superpuestas (más ancha = más dispersión)
- **📈 classification.png** — Gráfico 2D mostrando qué tan bien se separan los materiales
- **📋 predictions.csv** — Números: θ₀, pérdida de energía, X₀ para cada configuración
- **📝 SUMMARY.md** — Reporte completo incluyendo cuántos eventos necesitás para distinguir cada par

### Ideas para investigaciones

| Dificultad | Investigación | Qué cambiar |
|-----------|--------------|-------------|
| ⭐ | ¿Podemos distinguir PVC de PE? | Simulá PE + PVC, mirá las distribuciones |
| ⭐ | ¿Importa la energía del haz? | Mismos materiales a [1.0, 3.0, 6.0, 10.0] GeV/c |
| ⭐⭐ | Blancos gruesos vs finos | Mismo material a 5, 10, 20 mm de espesor |
| ⭐⭐ | Materiales arqueológicos | Simulá obsidiana, cuarzo, calcita, Fe₂O₃ |
| ⭐⭐⭐ | El desafío PE vs PP | ¿Hay ALGÚN ajuste que los separe? (Pista: ¡es muy difícil!) |

### Solución de problemas

| Problema | Solución |
|---------|---------|
| La Action no corrió | Verificá que tu archivo termine en `.yaml` y esté en `requests/` (no en `requests/examples/`) |
| Error de YAML | Revisá la indentación (usá espacios, no tabs). Usá el [validador YAML](https://www.yamllint.com/) |
| Material desconocido | Revisá la lista de materiales al final de `template.yaml` |
| Los gráficos se ven raros | Revisá `thickness_mm` — ¿quisiste decir 10.0 mm o 1.0 cm? |

---

### Glosario / Glossary

| Español | English | Symbol |
|---------|---------|--------|
| Dispersión múltiple de Coulomb | Multiple Coulomb Scattering | MCS |
| Ángulo de dispersión | Scattering angle | θ₀ |
| Longitud de radiación | Radiation length | X₀ |
| Momento del haz | Beam momentum | p |
| Espesor del blanco | Target thickness | x |
| Pérdida de energía | Energy loss | ΔE |
