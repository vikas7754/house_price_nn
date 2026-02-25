import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time

import torch
import torch.fx
import torch.nn as nn

from core.data import get_dataloader
from core.model import HousePriceNN
from core.train_step import train_step
from tabulate import tabulate
from core.custom_op_model import HousePriceModelWithCustomOp

def run_fx():
    print("Running FX Symbolic Trace Execution...")
    dataloader = get_dataloader(batch_size=32)
    # Use the model with custom operator
    model = HousePriceModelWithCustomOp()

    # Symbolic Trace
    # FX creates a GraphModule that is often faster interpretable Python code
    traced_model = torch.fx.symbolic_trace(model)

    output_path_tabular = os.path.join(os.path.dirname(__file__), "../outs/fx_graph_tabular.txt")
    with open(output_path_tabular, "w") as f:
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in traced_model.graph.nodes]
        f.write(tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"]))
        print(f"Tabular graph saved to {output_path_tabular}")

    output_path_raw = os.path.join(os.path.dirname(__file__), "../outs/fx_graph_raw.txt")
    with open(output_path_raw, "w") as f:
        f.write(str(traced_model.graph))
        print(f"Raw graph saved to {output_path_raw}")

    output_path_code = os.path.join(os.path.dirname(__file__), "../outs/fx_generated_code.py")
    with open(output_path_code, "w") as f:
        f.write(traced_model.code)
        print(f"Generated code saved to {output_path_code}")


    optimizer = torch.optim.Adam(traced_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    start_time = time.time()
    steps = 0
    max_steps = 50

    traced_model.train()
    for _ in range(5):
        for x, y in dataloader:
            loss = train_step(traced_model, optimizer, x, y, loss_fn)
            steps += 1
            if steps % 10 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    end_time = time.time()
    print(f"FX Mode finished in {end_time - start_time:.4f}s")

    # Save model for UI
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/house_price_model_fx.pth')
    torch.save(traced_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    run_fx()
