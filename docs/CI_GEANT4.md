# Running Geant4 in GitHub Actions (advanced)

The default student loop uses the **Highland predictor** (fast, no Geant4 needed).

For real Geant4 runs you have three realistic options.

## Option A — Self-hosted runner in Córdoba (recommended)

**Why:** fastest, most reliable, no 30–60 minute Geant4 build in CI.

1. Pick a Linux machine (school PC / university workstation / small server).
2. Install Geant4 (and datasets) once.
3. Add the machine as a **self-hosted GitHub Actions runner**.
4. Add runner labels: `self-hosted` and `geant4`.

Then you can run the workflow:
- **Actions → “Geant4 simulation (advanced, optional)” → Run workflow**

The workflow will:
- build `simulation/` (BeamScan Geant4 app)
- generate macros from the student request YAML
- run Geant4 and upload CSV outputs as artifacts

## Option B — Prebuilt Geant4 Docker image

Maintain a Docker image that already contains Geant4 and datasets, then change the workflow:
- set `runs-on: ubuntu-latest`
- add `container: <your-image>`

This is good if you don’t want a self-hosted runner, but you must keep the image updated.

## Option C — Build Geant4 from source inside CI

This works, but is heavy (long build + large datasets). Only recommended if you use caching and are comfortable maintaining CI infrastructure.

