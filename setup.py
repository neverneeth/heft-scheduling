"""
Setup configuration for HEFT Scheduling Framework.

This makes the framework installable as a Python package using pip install.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'FRAMEWORK_README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="heft-scheduling-framework",
    version="1.0.0",
    description="A rigorous, extensible framework for DAG-based workflow scheduling algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HEFT Scheduling Framework Team",
    author_email="",
    url="https://github.com/yourusername/heft-scheduling-framework",
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.6.3",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="scheduling algorithms DAG workflow HEFT Q-learning",
    project_urls={
        "Documentation": "https://github.com/yourusername/heft-scheduling-framework",
        "Source": "https://github.com/yourusername/heft-scheduling-framework",
        "Bug Reports": "https://github.com/yourusername/heft-scheduling-framework/issues",
    },
)