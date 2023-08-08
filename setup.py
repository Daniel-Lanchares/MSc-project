# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:23:30 2023

@author: danie
"""

from setuptools import find_packages, setup
setup(
    name='CBC_estimator',
    packages=find_packages(), # include=['CBC_estimator'] might revert to explicit creation
    version='0.1.0',
    description='Implementation of an NPE approach to gravitational wave parameter estimation',
    author='Daniel Lanchares',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
