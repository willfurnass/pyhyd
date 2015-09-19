from setuptools import setup, find_packages
import os


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pyhyd',
    version='0.1.5',
    packages=find_packages(),

    install_requires=[
        "numpy >= 1.5.1",
        "scipy >= 0.11.0",
    ],

    author='Will Furnass',
    author_email='will@thearete.co.uk',
    description='Functions for calculating various hydraulic quantities ' +
                'relating to drinking water distribution networks',
    license='GPL 3.0',
    keywords='hydraulics drinking water distribution systems',
)
