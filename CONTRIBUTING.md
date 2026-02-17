# Contributing (students)

This repo is designed so students can contribute **without touching Geant4 code**.

## Quick contribution workflow

1. **Fork** the repo.
2. Create a new branch.
3. Copy the template request:
   - `requests/examples/template.yaml` → `requests/<yourname>_<topic>.yaml`
4. Edit your request file:
   - choose 2 beam momenta (e.g., `[3.0, 6.0]`)
   - choose materials + thickness
5. Open a **Pull Request**.

When you open the PR, GitHub Actions will:
- validate your YAML file
- run the Highland predictions
- generate plots + a markdown summary
- post the summary into the PR as a comment

You can download full-resolution plots from the Action run’s **Artifacts**.

## How to propose a new material

Two options:

### A) Use an existing Geant4/NIST material (recommended)
Set `geant4_name` to a known material (examples in `requests/examples/`).

### B) Prediction-only custom material (no code changes)
Add `x0_cm` and `rho` (g/cm³) in your request entry. This will work for Highland predictions.

> Note: running **Geant4 simulation** for a custom material requires that material to be defined in the Geant4 code.

## What makes a good student request

- Keep thickness fixed (e.g., 10 mm) so comparisons are meaningful.
- Compare 2–4 materials per request.
- Include at least one “anchor” material (e.g., PE or PVC) so different requests connect.

