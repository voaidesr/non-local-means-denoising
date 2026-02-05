# MCNLM Python Implementation

This package contains the code used to generate the figures and results in the paper, including:
- baseline Non-Local Means (NLM)
- Monte Carlo NLM (MCNLM)
- KD-Tree accelerated NLM
- Hashed NLM
- FFT-based noise estimation

## Requirements
- Python >= 3.11
- Poetry

## Setup
```bash
cd mcnlm

poetry install
```

Tip: the first run is slower because Numba compiles kernels.

## Reproduce Plots
All plots default to `docs/res` in the repo root.

```bash
# list available plot names
poetry run mcnlm --list

# generate every plot (reproducible)
poetry run mcnlm --all --deterministic

# generate a single plot by name
poetry run mcnlm --plot <plot_name> # (see from list)

# generate multiple plots
poetry run mcnlm --plot methods_comparison_clock --plot hashednlm

# custom output directory
poetry run mcnlm --plot noise_comparison_visual2 --out-dir /tmp/mcnlm-plots
```

## Resources
- Poetry: https://python-poetry.org/
- Numba: https://numba.pydata.org/
