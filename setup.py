from glob import glob
from setuptools import find_packages, setup

package_name = "befr_visual_tracking"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*")),
    ],
    install_requires=[
        "numpy>=1.24",
        "PyYAML>=6.0",
        "scipy>=1.10",
        "setuptools",
    ],
    zip_safe=True,
    maintainer="BEFR Team",
    maintainer_email="team@example.com",
    description="Multi-camera Bayesian visual tracking for quadcopter state estimation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_simulator = befr_visual_tracking.camera_simulator_node:main",
            "dataset_recorder = befr_visual_tracking.dataset_recorder_node:main",
            "generate_canonical_dataset = befr_visual_tracking.generate_canonical_dataset:main",
            "generate_calibration_datasets = befr_visual_tracking.generate_calibration_datasets:main",
            "run_calibration = befr_visual_tracking.run_calibration:main",
            "run_experiments = befr_visual_tracking.run_experiments:main",
            "visual_tracker = befr_visual_tracking.tracker_node:main",
        ],
    },
)
