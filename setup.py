# -*- coding: utf-8 -*-
''' In case torch installation didn't update correctly
Successfully installed
cmake-3.27.2
lit-16.0.6
nvidia-cublas-cu11-11.10.3.66
nvidia-cuda-cupti-cu11-11.7.101
nvidia-cuda-nvrtc-cu11-11.7.99
nvidia-cuda-runtime-cu11-11.7.99
nvidia-cudnn-cu11-8.5.0.96
nvidia-cufft-cu11-10.9.0.58
nvidia-curand-cu11-10.2.10.91
nvidia-cusolver-cu11-11.4.0.1
nvidia-cusparse-cu11-11.7.4.91
nvidia-nccl-cu11-2.14.3
nvidia-nvtx-cu11-11.7.91
torch-2.0.1
torchvision-0.15.2 triton-2.0.0
'''

from setuptools import find_packages, setup


def get_requirements(kind=None):
    # Straight from bilby. May allow separate installation
    if kind is None:
        filename = "requirements.txt"
    else:
        filename = f"{kind}_requirements.txt"
    with open(filename, "r") as file:
        requirements = file.readlines()
    return requirements


setup(
    name='dtempest',
    packages=find_packages(),  # include=['dtempest'] might revert to explicit creation
    version='0.1.0',
    description='Implementation of an NPE approach to gravitational wave parameter estimation',
    author='Daniel Lanchares',
    license='MIT',
    python_requires="~=3.10",
    install_requires=get_requirements(),
    # In case it can be separated into lal_dependent and not
    # extras_require={
    #     "gw": get_requirements("gw"),
    #     "mcmc": get_requirements("mcmc"),
    #     "all": (
    #             get_requirements("sampler")
    #             + get_requirements("gw")
    #             + get_requirements("mcmc")
    #             + get_requirements("optional")
    #     ),
    # },

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
