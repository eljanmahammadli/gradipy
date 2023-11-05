from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="gradipy",
    version="0.1.0",
    description="A Lightweight Neural Network Library only using NumPy with Pytorch-like API",
    packages=["gradipy", "gradipy.nn"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eljanmahammadli/gradipy",
    author="Eljan Mahammadli",
    author_email="eljanmahammadlI@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=1.26.1"],
    extras_require={
        "dev": ["pytest>=7.4.3", "twine>=4.0.2", "torch>=2.1.0"],
    },
    python_requires=">=3.8",
)
