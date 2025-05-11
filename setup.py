from setuptools import setup, find_packages

setup(
    name="eqrpy",
    version="0.1.0",
    description="Algorithms for computing equilibria of normal-form games",
    author="Runzhe Yang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pplpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)