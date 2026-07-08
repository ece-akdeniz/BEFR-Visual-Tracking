from setuptools import find_packages, setup

setup(
    name="befr_visual_tracking",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy>=1.24",
        "PyYAML>=6.0",
        "scipy>=1.10",
    ],
    entry_points={
        "console_scripts": [
            "generate_canonical_dataset = befr_visual_tracking.generate_canonical_dataset:main",
            "generate_calibration_datasets = befr_visual_tracking.generate_calibration_datasets:main",
            "run_calibration = befr_visual_tracking.run_calibration:main",
            "run_experiments = befr_visual_tracking.run_experiments:main",
        ],
    },
)
