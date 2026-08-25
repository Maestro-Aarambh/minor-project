"""
Download EMBER2024 raw feature archives into data/raw/.

Why it exists:
Reproducibility. Anyone can rebuild the exact raw inputs used in the paper
by re-running this module with the same file_type / split arguments.
"""

from __future__ import annotations

import os
from pathlib import Path

import thrember


def download_ember(
    download_dir: str | Path,
    *,
    file_type: str = "Dot_Net",
    include_challenge: bool = True,
) -> Path:
    """
    Download train + test for one file type, plus the evasive challenge set.

    Default file_type is Dot_Net for the first runnable baseline: Win32/Win64
    archives are 15–29 GB and are awkward on a 16 GB laptop. Swap to Win64 or
    Win32 in the YAML config when more disk/RAM is available — same code path.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    try:
        thrember.download_dataset(str(download_dir), split="train", file_type=file_type)
        thrember.download_dataset(str(download_dir), split="test", file_type=file_type)
        if include_challenge:
            thrember.download_dataset(str(download_dir), split="challenge")
    finally:
        os.chdir(cwd)

    return download_dir
