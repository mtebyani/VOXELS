from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_desc = f.read()

setup(
    name='POM',
    version='0.0.1',
    description='Planes Of Motion (POM) model of voxel robots',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/mtebyani/VOXELS',
    author='Maryam Tebyani & Alex Spaeth',
    classifiers=[
        'Development Status :: 2 - Beta',
        'Programming Language :: Python :: 3'],
    packages=find_packages(exclude=()),
    install_requires=['numpy'])
