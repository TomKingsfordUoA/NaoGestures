#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f_requirements:
    requirements = f_requirements.read().splitlines()

setup(
    name='nao-gestures',
    version='1.0',
    description='Makes it easier to realise gestures on a Nao robot.',
    author='Tom Kingsford',
    author_email='tkin063@aucklanduni.ac.nz',
    url='',
    python_requires="==2.7.*",
    install_requires=requirements,
    packages=['nao_gestures', 'pymo'],
    entry_points={
        'console_scripts': ['nao_gestures=nao_gestures.cli:main']
    },
)
