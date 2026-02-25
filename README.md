# House Price Prediction & PyTorch Execution Modes

This project compares different PyTorch execution modes (Eager, `torch.compile`, FX, TorchScript) using a house price prediction task.

## Prerequisties

Ideally use `uv` for package management, but `pip` works too.

```bash
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

## Structure

- **core/**: Common logic (Model, Data, Training Step).
- **run/**: Scripts to train the model using different modes.
- **profiling/**: Scripts to profile execution and generate chrome traces.
- **ui/**: Streamlit app for inference.

## Usage

### 1. Generate Data

First, create the dummy dataset:

```bash
python generate_data.py
# Creates housing.csv
```

### 2. Run Training (Different Modes)

Run any of the scripts in `run/`. Note that `torch.compile` might be slower on the first run due to compilation overhead.

```bash
python run/eager.py        # Default PyTorch
python run/torch_compile.py # using torch.compile()
python run/fx.py           # using torch.fx.symbolic_trace()
python run/torchscript.py  # using torch.jit.script()
```

### 3. Profiling

Run the profiling scripts to generate Chrome traces in `profiling/traces/`.

```bash
python profiling/profile_eager.py
python profiling/profile_torch_compile.py
python profiling/profile_custom_op_fx.py
```

### 4. Viewing Traces

1. Open Google Chrome.
2. Navigate to `chrome://tracing`.
3. Click "Load" and select a `.json` file from `profiling/traces/`.
4. Analyze the `model_train_step` blocks to see execution details.

### 5. Streamlit UI

Verify the model allows for predictions.

```bash
streamlit run ui/app.py
```

## Explanation of Modes

- **Eager**: Standard Python execution. Flexible but slower due to interpreter overhead.
- **TorchCompile**: JIT compiles PyTorch code into optimized kernels (using Inductor backend by default).
- **FX**: Symbolic tracing to capture the graph. Useful for transformations and some optimizations.
- **TorchScript**: Statistical compilation to an intermediate representation (IR) that can run in C++ or other environments.

## Note on Accuracy

This project focuses on **system performance and execution flow**, not model accuracy. A simple Linear Regression might beat this NN on such small data, but the goal is to profile NN execution.
