#!/usr/bin/env python

from distutils.core import setup

setup(
    name='nao-gestures',
    version='1.0',
    description='Makes it easier to realise gestures on a Nao robot.',
    author='Tom Kingsford',
    author_email='tkin063@aucklanduni.ac.nz',
    url='',
    python_requires="==2.7.*",
    install_requires=[
        "pandas==0.24.2",
        "scikit-learn==0.20.4",
        # "naoqi",
    ],
    packages=['nao_gestures', 'pymo'],
)
