# Geant4 CI -- How It Works

## Overview

The Geant4 simulation runs on **GitHub-hosted runners** using the
[JeffersonLab Docker image](https://hub.docker.com/r/jeffersonlab/geant4)
(`jeffersonlab/geant4:g4v11.3.2-fedora40`). This image includes:

- Geant4 11.3.2 (pre-compiled)
- All required datasets (G4NDL, G4EMLOW, etc.)
- cmake, gcc, Python 3

No self-hosted runner needed. No Geant4 installation needed.

## How to trigger

1. Go to **Actions** > **Geant4 Simulation**
2. Click **Run workflow**
3. Enter the request file path (e.g. `requests/full_classification.yaml`)
4. Optionally set events per config (default: 2000)
5. Click **Run workflow**

The job will:
- Pull the Docker image (~7 GB, cached after first run)
- Build BeamScan from source (~2 min)
- Run all material/momentum combinations
- Compare Geant4 results with Highland predictions
- Upload everything as downloadable artifacts

## Typical run times

| Config | Events | Approx. time |
|--------|--------|--------------|
| 1 material, 1 momentum, 2000 events | 2000 | ~5 min |
| 12 materials, 2 momenta, 2000 events | 48000 | ~20 min |
| 12 materials, 2 momenta, 10000 events | 240000 | ~60 min |

GitHub Actions free tier gives 2000 min/month. Use 2000 events for testing,
10000+ for final validation.

## Artifacts

After the run, download from the workflow run page:
- `geant4_output/` -- raw CSV event data per material
- `comparison/highland/` -- Highland predictions for comparison
- `comparison/geant4_vs_highland/` -- overlay plots
- `macros_auto/` -- generated Geant4 macros (reproducibility)
