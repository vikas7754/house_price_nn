from core.model import HousePriceNN
import torch.nn as nn
import torch
import sys
import os
import glob

# Load custom operators
# Assuming the .so files are in ../custom_op/ (where we built them in-place)
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../custom_op"))
sys.path.append(lib_path)

for lib_name in ["my_custom_op", "custom_mul"]:
    so_files = glob.glob(os.path.join(lib_path, f"{lib_name}*.so"))
    if so_files:
        torch.ops.load_library(so_files[0])
        print(f"Loaded custom op library from {so_files[0]}")
    else:
        print(f"Warning: {lib_name} library not found!")


class HousePriceModelWithCustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = HousePriceNN()
        self.scale_tensor = nn.Parameter(torch.ones(1))
        self.bias_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.base_model(x)
        # Scale the output using custom_mul, then add bias using custom_add
        out = torch.ops.my_mul_ops.custom_mul(out, self.scale_tensor)
        return torch.ops.my_ops.custom_add(out, self.bias_tensor)