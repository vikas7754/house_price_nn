from core.model import HousePriceNN
import torch.nn as nn
import torch
import sys
import os
import glob

# Load custom operator
# Assuming the .so file is in ../custom_op/ (where we built it in-place)
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../custom_op"))
sys.path.append(lib_path)
so_files = glob.glob(os.path.join(lib_path, "my_custom_op*.so"))
if so_files:
    torch.ops.load_library(so_files[0])
    print(f"Loaded custom op library from {so_files[0]}")
else:
    print("Warning: Custom op library not found!")


class HousePriceModelWithCustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = HousePriceNN()
        self.bias_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.base_model(x)
        # Use the custom operator here
        # Adding a learnable bias using custom_add
        return torch.ops.my_ops.custom_add(out, self.bias_tensor)