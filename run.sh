#!/bin/bash
set -e  # Exit strictly on error

echo "===================================================="
echo "🚀 Starting Project Execution Pipeline"
echo "===================================================="

# 1. Training (Execution Modes)
echo ""
echo "--- 1. Running Training Scripts ---"
echo "Running Eager Mode..."
uv run python run/eager.py

echo "Running torch.compile Mode..."
uv run python run/torch_compile.py

echo "Running FX Mode..."
uv run python run/fx.py

echo "Running TorchScript Mode..."
uv run python run/torchscript.py

# 2. Profiling
echo ""
echo "--- 2. Running Profiling Scripts ---"
echo "Profiling Eager Mode..."
uv run python profiling/profile_eager.py

echo "Profiling torch.compile Mode..."
uv run python profiling/profile_torch_compile.py

echo "Profiling FX Mode..."
uv run python profiling/profile_fx.py

echo "Profiling TorchScript Mode..."
uv run python profiling/profile_torchscript.py

# 3. Benchmarking
echo ""
echo "--- 3. Running Benchmarks ---"
uv run python benchmark_runtime.py

echo ""
echo "===================================================="
echo "✅ All Scripts Executed Successfully!"
echo "===================================================="
echo "Next Steps:"
echo "1. View Chrome traces in profiling/traces/"
echo "2. Run the UI: uv run streamlit run ui/app.py"
echo "===================================================="
