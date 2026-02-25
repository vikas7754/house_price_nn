import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step
from core.device import get_device

def run_torch_compile():
    device = get_device()
    print(f"Running torch.compile Execution on {device}...")
    dataloader = get_dataloader(batch_size=32, pin_memory=(device.type == 'cuda'))
    model = HousePriceNN().to(device)

    # Compile the model
    # backend='inductor' is the default and usually best for CPU
    compiled_model = torch.compile(model)

    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    steps = 0
    max_steps = 50

    compiled_model.train()
    for _ in range(5):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # Note: compilation happens on the first execution
            loss = train_step(compiled_model, optimizer, x, y, loss_fn)
            steps += 1
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"torch.compile Mode finished in {end_time - start_time:.4f}s")

    # Save model for UI
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/house_price_model_torch_compile.pth')
    torch.save(compiled_model._orig_mod.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    run_torch_compile()
