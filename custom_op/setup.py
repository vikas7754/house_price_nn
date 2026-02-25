from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='my_custom_op',
    ext_modules=[
        CppExtension('my_custom_op', ['my_custom_op.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
