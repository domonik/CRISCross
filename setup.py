from setuptools import setup, find_packages

setup(
    name="CRISCross",
    version="0.1.0",
    description="Minimal package for running CRISCross models",
    author="domonik",
    packages=find_packages(
        include=[
            "CRISCross",
        ],
    ),
    include_package_data=True,
    package_data={
        "": ["**/*.pkl"],
    },
    install_requires=[
        "torch",
        "numpy",
        "huggingface_hub",
        "pytorch_lightning",
        "pyBigWig",
        "biopython",
        "pandas",
        "alphagenome"
    ],
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "criscross-download-ag=CRISCross.downloadWholeAGTrack:main",
        ],
    },
)