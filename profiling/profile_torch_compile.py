import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step

def profile_torch_compile():
    print("Profiling torch.compile Execution...")
    dataloader = get_dataloader(batch_size=32)
    model = HousePriceNN()
    compiled_model = torch.compile(model)
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    compiled_model.train()
    
    # Warmup (important for compile to actually compile)
    print("Warmup...")
    iter_loader = iter(dataloader)
    for _ in range(5):
        x, y = next(iter_loader)
        train_step(compiled_model, optimizer, x, y, loss_fn)

    print("Starting Profiler...")
    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        ) as prof:
            for i in range(20):
                with record_function("model_train_step"):
                    try:
                        x, y = next(iter_loader)
                    except StopIteration:
                        iter_loader = iter(dataloader)
                        x, y = next(iter_loader)
                    
                    train_step(compiled_model, optimizer, x, y, loss_fn)
    except Exception as e:
        print(f"Profiling error: {e}")
        return

    trace_path = os.path.join(os.path.dirname(__file__), 'traces', 'trace_torch_compile.json')
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")

if __name__ == "__main__":
    profile_torch_compile()
