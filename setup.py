from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setup(
    name="bosk",
    version="0.1.0",
    author="NTAILab",
    description="Deep Forest package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"),
    license='MIT',
    python_requires='>=3.9',
)
