#!/usr/bin/env python3
import os.path
from setuptools import setup, find_packages

# Copyright 2017-2020 Satellogic SA.


# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("orbit_predictor", "version.py")) as fp:
    exec(fp.read(), version)


setup(
    name='orbit-predictor',
    version=version["__version__"],
    author='Satellogic SA',
    author_email='oss@satellogic.com',
    description='Python library to propagate satellite orbits.',
    long_description=open('README.rst').read(),
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    url='https://github.com/satellogic/orbit-predictor',
    # Keywords to get found easily on PyPI results,etc.
    keywords="orbit, sgp4, TLE, space, satellites",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'numpy>=1.8.2',
        'sgp4',
        'requests',
    ],
    extras_require={
        "fast": [
            "numba>=0.38",
            "scipy>=0.16",
        ],
        "dev": [
            "hypothesis",
            "flake8",
            "hypothesis[datetime]",
            "mock",
            "logassert",
            "pytest",
            "pytest-cov",
            "pytest-benchmark",
            "pytz",
        ],
    },
    python_requires=">=3.4",
)
