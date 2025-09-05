#!/usr/bin/env python3
"""
Setup script for the Power Ranking package.

This allows the power_ranking module to be installed and imported properly.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="power_ranking",
    version="1.0.0",
    description="NFL Power Rankings Calculator",
    author="Power Ranking System",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'power-ranking=power_ranking.cli:main',
        ],
    },
    package_data={
        'power_ranking': ['config.yaml'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
