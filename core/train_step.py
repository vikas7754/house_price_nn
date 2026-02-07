import torch

def train_step(model, optimizer, x, y, loss_fn):
    """
    Performs a single training step:
    1. Zero gradients
    2. Forward pass
    3. Compute loss
    4. Backward pass
    5. Optimizer step
    """
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss
