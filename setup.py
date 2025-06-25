
from setuptools import setup, find_packages

# TODO: impliment a fix on this
setup(
    name="franka",
    version="0.1.0",
    description="A brief description of your package",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "some_dependency>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
