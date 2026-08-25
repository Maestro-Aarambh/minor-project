# Data directory

Contents of this directory are **gitignored** — datasets are large and must be obtained separately.

## Layout

```
data/
├── raw/         EMBER2024 downloads, as distributed (untouched)
└── processed/   Extracted representations ready for training:
                 byte sequences, opcode sequences, static feature vectors
```

## Obtaining EMBER2024

1. EMBER2024 is published by the FutureComputing4AI group:
   https://github.com/FutureComputing4AI/EMBER2024
2. This project uses only the **Win32/Win64 PE subset**, which includes:
   - The standard temporal train/test split (weeks 1–64)
   - The **evasive challenge set** (~6,300 samples that initially evaded
     every AV engine on VirusTotal) — the key evaluation target
3. Download the archives into `data/raw/` and run the Stage 1 pipeline
   (`src/data/`) to populate `data/processed/`.

## Note on malware safety

EMBER2024 distributes **extracted features and metadata**, not live binaries,
for most tasks. If any raw PE samples are handled (e.g. for disassembly),
treat them as live malware: work inside an isolated environment and never
execute samples.
