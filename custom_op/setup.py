import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ext_modules = [
    CppExtension('my_custom_op', ['my_custom_op.cpp']),
]

if torch.cuda.is_available():
    ext_modules.append(CUDAExtension('custom_mul', ['custom_mul.cpp']))
else:
    ext_modules.append(CppExtension('custom_mul', ['custom_mul.cpp']))

setup(
    name='my_custom_op',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })
