from setuptools import setup, find_packages

setup(
    name="ffp",
    version="1.42",
    description="A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP)",
    long_description="""A Python implementation of the FFP (Flux Footprint Prediction) model. 
    This package provides functions to calculate flux footprints and footprint climatologies 
    based on the parameterisation described in Kljun et al. (2015).""",
    author="Natascha Kljun",
    packages=find_packages(),  # Automatically detects the 'ffp' directory as a package
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",  # Required for plotting functionality
    ],
    python_requires=">=2.7.5",
)
