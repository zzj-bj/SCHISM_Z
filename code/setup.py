from setuptools import find_packages, setup

setup(
    name="schism",
    version="0.0.1",
    author="Florent Brondolo",
    author_email="florent.brondolo@akkodis.com",
    description="Simple Deep Learning library for geoscience and vision",
    packages_dir={"": "classes"},
    packages=find_packages(where="classes"),
    install_requires=[
        "tk",
        "torchmetrics>=1.5.2",
        "numpy>=1.17.4",
        "torch>=2.3.0",
        "matplotlib",
        "tqdm",
        "torchvision",
        "patchify",
        "peft>=0.10.0",
        "transformers>=4.40.0",
        "bitsandbytes>=0.42.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <3.12",
)
