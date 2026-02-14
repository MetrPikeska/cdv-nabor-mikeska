from setuptools import setup, find_packages

setup(
    name='traffic_analysis',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'opencv-python',
        'torch'
    ],
)