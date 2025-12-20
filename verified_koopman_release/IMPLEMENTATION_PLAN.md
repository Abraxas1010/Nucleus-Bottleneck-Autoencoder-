# Verified Koopman Release Artifacts Plan

This directory is a clean release bundle intended to be copied into a fresh GitHub repo.

## Scope

Included:
- Minimal PyTorch implementation for Koopman AE + nucleus bottleneck + end-to-end Heyting variant
- Lean 4 proofs for the nucleus and Heyting operators
- Local scripts/configs/docs/tests for reproducing and validating claims

Excluded (intentionally):
- Internal SSH orchestration and remote GPU scripts
- Internal WIP notes and run directories outside this folder

## Local Validation Checklist

1. Lean verification:
   - `./scripts/verify_lean.sh`
2. Python install:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt && pip install -e .`
3. Smoke experiments:
   - `python scripts/run_experiments.py --experiment capability --epochs 20`
   - `python scripts/run_experiments.py --experiment lyapunov --epochs 20`
   - `python scripts/run_experiments.py --experiment heyting --epochs 20`
4. Tests:
   - `pytest -q`

## Publishing

You will manually copy this folder into a new repo and push it; no Git tooling is used here.

