import torch
import torch.nn as nn
import torch.fx as fx
import os

# Load the custom operator library
# This is platform specific, assuming linux/mac .so
# In a real package, you would import the package
# For this example, we load library directly if needed, or rely on torch.ops
# But we need to make sure the shared object is loaded.
# The setup.py install/develop makes it importable usually, but for local inplace build:
import sys
sys.path.append(os.getcwd())
try:
    import my_custom_op
    print("Imported my_custom_op successfully")
except ImportError:
    # If not installed as a module, we might need to load .so manually or just hope torch.ops works if we run script in same env after build.
    # Actually, simplest for this demo is to rely on the side-effect of importing the extension if it were a module.
    # Let's try to load it via torch.ops after ensuring it's loaded.
    pass

# We need to make sure the symbol is registered.
# If we built inplace, there should be a .so file we can load.
import glob
so_files = glob.glob("my_custom_op*.so")
if so_files:
    torch.ops.load_library(os.path.abspath(so_files[0]))
    print(f"Loaded library from {so_files[0]}")
else:
    print("Warning: Could not find .so file to load explicitly. Assuming it's already loaded or installed.")


class MyModule(nn.Module):
    def forward(self, x, y):
        return torch.ops.my_ops.custom_add(x, y)

def test():
    m = MyModule()
    x = torch.randn(5)
    y = torch.randn(5)
    
    # Eager execution
    out = m(x, y)
    print("Eager Output:", out)
    
    # FX Tracing
    # By default, custom ops registered via torch.ops are traceable!
    traced = fx.symbolic_trace(m)
    print("\nFO Graph Code:")
    print(traced.code)
    
    # Run traced module
    out_traced = traced(x, y)
    print("\nTraced Output:", out_traced)
    
    assert torch.allclose(out, out_traced)
    print("\nVerification Successful!")

    # Print the graph to see the node target
    print("\nGraph Nodes:")
    for node in traced.graph.nodes:
        print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}")

if __name__ == "__main__":
    test()
