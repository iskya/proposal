# ═══════════════════════════════════════════════════════════════
# BeamScan — Docker image for local and CI reproducibility
# ═══════════════════════════════════════════════════════════════
#
# Usage (Highland predictions only — no Geant4 needed):
#   docker build -t beamscan .
#   docker run --rm -v $PWD:/work beamscan \
#     python analysis/highland_calculator.py requests/full_classification.yaml
#
# Usage (with Geant4 — uncomment the base image below):
#   docker build -t beamscan-g4 .
#   docker run --rm -v $PWD:/work beamscan-g4 \
#     bash scripts/run_all.sh
# ═══════════════════════════════════════════════════════════════

# ── Option A: Lightweight (Highland predictions only) ──
FROM python:3.11-slim

# ── Option B: Full Geant4 (uncomment and comment Option A) ──
# FROM geant4/geant4:11.2.2
# RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /work

# Install Python dependencies
COPY analysis/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy analysis tools (simulation code is mounted at runtime)
COPY analysis/ /work/analysis/
COPY scripts/ /work/scripts/
COPY data/ /work/data/

# Default: run the full classification
CMD ["python", "analysis/highland_calculator.py", "requests/full_classification.yaml"]
