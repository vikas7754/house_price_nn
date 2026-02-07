import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import time
from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step

def run_eager():
    print("Running Eager Execution...")
    dataloader = get_dataloader(batch_size=32)
    model = HousePriceNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    start_time = time.time()
    steps = 0
    max_steps = 1000  # Increased for better accuracy

    model.train()
    for epoch in range(50):  # Increased epochs
        for x, y in dataloader:
            loss = train_step(model, optimizer, x, y, loss_fn)
            steps += 1
            if steps % 100 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
            
    end_time = time.time()
    print(f"Eager Mode finished in {end_time - start_time:.4f}s")
    
    # Save model for UI
    torch.save(model.state_dict(), 'house_price_model.pth')
    print("Model saved to house_price_model.pth")

if __name__ == "__main__":
    run_eager()
