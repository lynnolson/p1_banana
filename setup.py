#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='P1 Banana',
      version='0.0.0',
      description='Banana learning agent',
      packages=find_packages(),
      install_requires = required,
     )
