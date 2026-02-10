# 🏡 House Price Prediction Project

This document encapsulates the technical details, architecture, and advanced PyTorch concepts demonstrated in this project. Use this as a reference for discussing the project in interviews.

---

## 1. Project Overview

**Goal:** Build a production-grade ML pipeline to predict house prices while demonstrating and benchmarking advanced PyTorch execution modes.
**Core Value:** Unlike a standard "model.fit()" project, this project focuses on **ML Systems Engineering**—optimizing how the model runs, profiling its execution, and understanding the trade-offs between flexibility (Eager) and performance (Compiled).

---

## 2. Tech Stack & Tools

- **Framework:** PyTorch (v2.x)
- **Data Processing:** Pandas, NumPy
- **Visualization/UI:** Streamlit
- **Profiling:** `torch.profiler` (exporting to Chrome Tracing `chrome://tracing`)
- **Package Management:** `uv` (modern, fast Python package manager)

---

## 3. Data Pipeline & Preprocessing

**Input Features (5):** Income, House Age, Rooms, Bedrooms, Population.
**Target:** Price.

### Key Engineering Decision: Global Normalization

- **Problem:** Neural networks struggle with unscaled data (e.g., Income ~60k vs Age ~5).
- **Solution:** applied **Z-Score Normalization** (StandardScaler) to both features AND targets.
  $$x' = \frac{x - \mu}{\sigma}$$
- **Artifacts:** We save `training_stats.pt` typically containing $\mu$ (mean) and $\sigma$ (std) during training.
- **Inference:** The UI loads these stats to ensuring input data is normalized exactly like the training data, and the output price is correctly denormalized back to dollar amounts.

---

## 4. Model Architecture

**Type:** Feed-Forward Neural Network (Multi-Layer Perceptron)
**Structure:**

1. **Input Layer:** 5 neurons
2. **Hidden Layer 1:** 256 neurons + ReLU (Non-linearity)
3. **Hidden Layer 2:** 256 neurons + ReLU
4. **Output Layer:** 1 neuron (Linear/Regression output)

**Training Config:**

- **Optimizer:** Adam (Adaptive learning rates)
- **Loss Function:** MSE (Mean Squared Error), standard for regression.
- **Batch Size:** 2048 (Large batch size chosen to saturate CPU/Compiler).

---

## 5. Advanced PyTorch Concepts (The "Star" of the Show)

This is the most critical section for an ML Engineer interview.

### A. Eager Execution (Standard PyTorch)

- **What it is:** Default interpretation. Python runs line-by-line.
- **Pros:** Extremely debuggable, dynamic (can use `if` statements, print inside model).
- **Cons:** Slower. Python interpreter overhead (GIL) limits performance, especially with small ops.
- **Trace Analysis:** You see gaps between kernel launches on the timeline due to Python overhead.

### B. TorchScript (`torch.jit.script`)

- **What it is:** A legacy JIT compiler that converts PyTorch code into an Intermediate Representation (IR).
- **Why use it:** Serialization. You can save a model and load it in C++ (production) execution without Python installed.
- **Pros:** Portable, decent speedup.
- **Cons:** Rigid. Doesn't support all Python features. Requires rewriting code to be "scriptable".

### C. FX (`torch.fx`)

- **What it is:** A toolkit for **symbolic tracing**. It traces the model execution to produce a generic Python graph (`GraphModule`).
- **Why use it:** Graph transformations. It's the foundation for quantization (low precision), operator fusion, and custom optimizations.
- **How it works:** It feeds "fake" data to record operations into a graph, then regenerates simplified Python code.

### D. `torch.compile` (PyTorch 2.0+)

- **What it is:** The modern compiler uses **TorchDynamo** (graph capture) and **TorchInductor** (backend code generator).
- **How it works:**
  1. Captures the generic graph (like FX).
  2. Compiles it into highly optimized Triton/C++ kernels.
  3. Performs advanced fusion (e.g., combining `Linear` + `ReLU` into one kernel read/write).
- **Performance:** Usually the fastest option (~2x-3x speedup on GPU, perceptible on CPU).

---

## 6. Performance Engineering & "Gotchas"

### The "Dynamic Shape" Problem

**Observation:** Initially, `torch.compile` was slower (129ms) than Eager (8ms).
**Cause:** The compiler specializes kernels for specific input shapes (Batch Size 2048). The last batch of the epoch had a different size (residual), forcing a **Re-Compilation**.
**Fix:** Set `drop_last=True` in DataLoader.
**Result:**

- **Eager:** ~8ms
- **torch.compile:** ~3-4ms (**2x Speedup**)

This demonstrates deep understanding of how JIT compilers work (static vs dynamic shapes).

---

## 7. Results Dashboard

We built a Streamlit UI to visualize:

1. **Real-time Inference:** Entering values -> Normalization -> Model -> Denormalization.
2. **Offline Benchmarks:** Visual comparison of the 4 execution modes.
3. **Debug Inspection:** Viewing raw normalized tensors vs physical values.
