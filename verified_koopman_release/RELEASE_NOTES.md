# Release Artifact Notes (You Will Publish Manually)

This folder is a **self-contained release artifact** prepared for a new GitHub repository.

- You (human) will manually create the GitHub repo and push these files.
- This workspace repo’s internal infrastructure is intentionally **not** included (e.g. SSH orchestration, internal WIP notes).
- No GitHub Actions workflows are included.

## What’s Included

- `lean/`: Lean 4 + Mathlib project with the four core verified files:
  - `VerifiedKoopman/NucleusReLU.lean`
  - `VerifiedKoopman/NucleusThreshold.lean`
  - `VerifiedKoopman/HeytingOps.lean`
  - `VerifiedKoopman/ParametricHeyting.lean`
- `src/verified_koopman/`: minimal Python package implementing:
  - baseline Koopman AE
  - nucleus-bottleneck AE
  - bounded Heyting ops + learnable parameters
  - curriculum schedules + losses (for the “decoupled/curriculum” variant)
- `scripts/`: local-only entrypoints to verify Lean, train, and analyze.
- `tests/`: lightweight unit tests.
- `docs/`: installation + reproduction guide.

## What’s Excluded (by design)

- SSH scripts / remote orchestration (hardware-specific)
- Internal WIP experiment notes and run directories
- Any proprietary datasets

