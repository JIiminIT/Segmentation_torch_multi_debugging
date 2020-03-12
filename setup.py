"""Tinycat 설치 스크립트

run:
    > python setup.py install
"""
# Copyright 2020 Neurophet Inc. All Rights Reserved.
# Author: Daun Jung (iam@nyanye.com / djjung@neurophet.com)
import os
import io
from setuptools import find_packages
from setuptools import setup


# 2020-03-12 comment:
# Aqua Engine 1.3.0 및 SegEngine 2.1.7 버젼에서 배포되고 있는 dependency들의 버젼:
# numpy==1.16.2
# nibabel==2.5.0
# tensorflow-gpu==1.14.0
REQUIRED_PACKAGES = [
    "numpy", # 1.8.1
    "scipy", # 1.4.1
    "matplotlib",
    "nibabel",
    "joblib",
    "pandas",
    "click",
    "tqdm",
    "SimpleITK",
]

# tensorflow-gpu 패키지가 설치되어 있는 경우 그대로 사용
# pylint: disable=unused-import
try:
    import tensorflow as _test_
except ImportError:
    REQUIRED_PACKAGES.append("tensorflow")


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
