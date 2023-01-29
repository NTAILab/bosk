from setuptools import setup, find_packages

import bosk

with open("README.md", "r") as fh:
    long_description = fh.read()

# requirements = []

setup(
    install_requires=[],
    name="bosk",
    version=bosk.__version__,
    author="NTAILab",
    description="Deep Forest package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"),
    license='MIT',
    python_requires='>=3.9',
)
