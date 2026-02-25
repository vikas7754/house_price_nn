# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch performance benchmarking project that trains a feed-forward neural network (5→256→256→1 MLP) on house price prediction data while comparing four execution modes: Eager, torch.compile, FX symbolic tracing, and TorchScript. The focus is on **ML systems engineering**—profiling, benchmarking, and understanding execution trade-offs—not model accuracy.

## Commands

```bash
# Install dependencies (uv preferred)
uv pip install -r requirements.txt

# Generate synthetic dataset (creates housing.csv)
python generate_data.py

# Train individual modes
python run/eager.py
python run/torch_compile.py
python run/fx.py
python run/torchscript.py

# Run profiling (generates Chrome traces in profiling/traces/)
python profiling/profile_eager.py
python profiling/profile_torch_compile.py
python profiling/profile_fx.py
python profiling/profile_torchscript.py
python profiling/profile_custom_op_fx.py

# Run benchmarks (generates ui/runtime_metrics.json)
python benchmark_runtime.py

# Full pipeline (all training, profiling, and benchmarks)
bash run.sh

# Launch Streamlit UI
streamlit run ui/app.py

# Build custom C++ operator
cd custom_op && python setup.py build_ext --inplace
```

There are no test or lint commands configured.

## Architecture

### Core (`core/`)
Shared logic used by all execution modes:
- **model.py**: `HousePriceNN` — 5→256→256→1 with ReLU activations
- **data.py**: `HousePriceDataset` — loads CSV, applies Z-score normalization to features and targets, saves stats to `training_stats.pt`
- **train_step.py**: Standard training loop function (zero_grad → forward → MSE loss → backward → step)
- **custom_op_model.py**: `HousePriceModelWithCustomOp` — wraps `HousePriceNN` with a C++ custom operator for FX tracing compatibility

### Run Scripts (`run/`)
Each script loads data, instantiates/compiles the model for its mode, runs training, and saves weights to `models/`. Eager uses 1000 steps; compiled modes use 50 steps. All use `drop_last=True` in DataLoader to avoid dynamic shape recompilation with torch.compile.

### Profiling (`profiling/`)
Each script mirrors its run/ counterpart but wraps training in `torch.profiler` to export Chrome trace JSON files to `profiling/traces/`. Traces are viewed at `chrome://tracing`.

### Custom Operator (`custom_op/`)
C++ extension (`my_custom_op.cpp`) implementing a simple `custom_add` op with profiler recording, built via PyTorch's `CppExtension` in `setup.py`.

### UI (`ui/app.py`)
Streamlit app providing interactive inference (with input denormalization/normalization using `training_stats.pt`) and a runtime comparison dashboard reading from `ui/runtime_metrics.json`.

### Key Artifacts
- `training_stats.pt`: Z-score normalization stats (feature/target mean and std) — critical for consistent inference
- `models/*.pth`: Trained weights per execution mode
- `ui/runtime_metrics.json`: Benchmark results consumed by Streamlit
- `outs/`: FX graph dumps (raw, tabular, generated code)

## Key Technical Details

- **Dynamic shape gotcha**: torch.compile specializes kernels for specific batch sizes. Residual (last) batches with different sizes trigger costly recompilation. Fixed by `drop_last=True` in DataLoader.
- **Z-score normalization** is applied to both features AND targets. The UI must load `training_stats.pt` to denormalize predictions back to dollar amounts.
- Scripts use `uv run python` in `run.sh` for environment consistency.
