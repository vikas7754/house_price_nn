import torch
import torch.nn as nn
import time
import json
import os
import torch.fx
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step

def measure_time(model, optimizer, dataloader, loss_fn, steps=20):
    # Warmup
    iter_loader = iter(dataloader)
    for _ in range(5):
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            x, y = next(iter_loader)
        train_step(model, optimizer, x, y, loss_fn)

    # Benchmark
    start_time = time.time()
    for _ in range(steps):
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            x, y = next(iter_loader)
        train_step(model, optimizer, x, y, loss_fn)
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / steps) * 1000
    return avg_time_ms

def benchmark():
    print("Running Benchmarks...")
    # Use drop_last=True to ensure constant batch size for torch.compile
    dataloader = get_dataloader(batch_size=2048, drop_last=True)
    loss_fn = nn.MSELoss()
    metrics = {}

    # 1. Eager
    print("- Benchmarking Eager...")
    model = HousePriceNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics["Eager"] = measure_time(model, optimizer, dataloader, loss_fn)

    # 2. torch.compile
    print("- Benchmarking torch.compile...")
    model = HousePriceNN()
    compiled_model = torch.compile(model)
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.001)
    # Trigger compilation with one step before warmup/benchmark just in case
    x_dummy, y_dummy = next(iter(dataloader))
    train_step(compiled_model, optimizer, x_dummy, y_dummy, loss_fn)
    metrics["torch.compile"] = measure_time(compiled_model, optimizer, dataloader, loss_fn)

    # 3. FX
    print("- Benchmarking FX...")
    model = HousePriceNN()
    traced_model = torch.fx.symbolic_trace(model)
    optimizer = torch.optim.Adam(traced_model.parameters(), lr=0.001)
    metrics["FX"] = measure_time(traced_model, optimizer, dataloader, loss_fn)

    # 4. TorchScript
    print("- Benchmarking TorchScript...")
    model = HousePriceNN()
    scripted_model = torch.jit.script(model)
    optimizer = torch.optim.Adam(scripted_model.parameters(), lr=0.001)
    metrics["TorchScript"] = measure_time(scripted_model, optimizer, dataloader, loss_fn)

    print("\nResults (ms/step):")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f} ms")

    output_path = os.path.join(os.path.dirname(__file__), 'ui', 'runtime_metrics.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    benchmark()
