# Copyright 2020 Neurophet Inc. All Rights Reserved.
# Author: Daun Jung (iam@nyanye.com / djjung@neurophet.com)
from setuptools import find_packages
from setuptools import setup
import os
import io


REQUIRED_PACKAGES = [
    "numpy==1.18",
    "scipy==1.4.1",
    "tensorflow==2.1.0",
    "matplotlib",
    "nibabel",
    "joblib",
    "pandas",
    "click",
    "tqdm",
    "SimpleITK",
]

# Retrieve version from about.py
def get_version():
    about = {}
    root = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(root, "tinycat", "about.py"), encoding="utf-8") as f:
        # pylint: disable=exec-used
        exec(f.read(), about)

    return about


def setup_package():
    about = get_version()
    setup(
        name="tinycat",
        version=about["__version__"],
        author=about["__author__"],
        author_email=about["__author_email__"],
        description="tinycat is a neuroimaging and neural network package",
        install_requires=REQUIRED_PACKAGES,
        license="Copyright by NEUROPHET all rights reserved.",
        packages=find_packages(),
        package_data={"tinycat": ["resources/*.txt", "resources/example_mri.nii.gz"]},
        include_package_data=True,
        entry_points={
            "console_scripts": [
                "evaluate_segmentation = tinycat.app.evaluator:main",
                "to_nifti = tinycat.app.to_nifti:to_nifti_cmd",
                "aseg_to_label = tinycat.app.aseg_to_label:main",
                "surfer_mgz_to_nifti = tinycat.app.surfer_mgz_to_nifti:main",
                "tinycat = tinycat.__main__:main",
            ]
        },
        requires=[],
    )


if __name__ == "__main__":
    setup_package()
