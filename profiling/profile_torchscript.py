import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step
from core.device import get_device

def profile_torchscript():
    device = get_device()
    print(f"Profiling TorchScript Execution on {device}...")
    dataloader = get_dataloader(batch_size=32, pin_memory=(device.type == 'cuda'))
    model = HousePriceNN()
    scripted_model = torch.jit.script(model)
    scripted_model = scripted_model.to(device)
    optimizer = torch.optim.Adam(scripted_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    scripted_model.train()

    # Warmup
    print("Warmup...")
    iter_loader = iter(dataloader)
    for _ in range(5):
        x, y = next(iter_loader)
        x, y = x.to(device), y.to(device)
        train_step(scripted_model, optimizer, x, y, loss_fn)

    print("Starting Profiler...")
    try:
        with profile(
            activities=activities,
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
                    x, y = x.to(device), y.to(device)
                    train_step(scripted_model, optimizer, x, y, loss_fn)
    except Exception as e:
        print(f"Profiling error: {e}")
        return

    trace_path = os.path.join(os.path.dirname(__file__), 'traces', 'trace_torchscript.json')
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")

if __name__ == "__main__":
    profile_torchscript()
