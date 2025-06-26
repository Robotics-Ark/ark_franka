
from setuptools import setup, find_packages


setup(
    name="franka",
    version="0.1.0",
    description="Franka Panda robot driver for the ARK framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "ark",
        "franky",
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
