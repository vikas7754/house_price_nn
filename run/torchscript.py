import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step

def run_torchscript():
    print("Running TorchScript Execution...")
    dataloader = get_dataloader(batch_size=32)
    model = HousePriceNN()
    
    # Script the model
    # Jit Script compiles the model into an intermediate representation
    # that can be optimized and run independently of Python
    scripted_model = torch.jit.script(model)
    
    optimizer = torch.optim.Adam(scripted_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    start_time = time.time()
    steps = 0
    max_steps = 50

    scripted_model.train()
    for _ in range(5):
        for x, y in dataloader:
            loss = train_step(scripted_model, optimizer, x, y, loss_fn)
            steps += 1
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
            
    end_time = time.time()
    print(f"TorchScript Mode finished in {end_time - start_time:.4f}s")

if __name__ == "__main__":
    run_torchscript()
