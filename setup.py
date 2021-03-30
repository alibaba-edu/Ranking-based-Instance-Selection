
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension

requirements = ["torch", "torchvision"]

setup(
    name="ret_benchmark",
    version="3.0",
    author="",
    url="",
    description="PRISM",
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)