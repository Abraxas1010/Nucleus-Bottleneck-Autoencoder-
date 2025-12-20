from __future__ import annotations

from pathlib import Path

__all__ = [
    "run_lean_gate",
]


def run_lean_gate(*, lean_dir: Path) -> None:
    from verified_koopman.verification.lean_gate import run

    run(lean_dir=lean_dir)
