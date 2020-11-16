import setuptools, os

PACKAGE_NAME = 'forex-utils'
VERSION = '0.0.1'
AUTHOR = 'Tim Esler'
EMAIL = 'tim@gmail.com'
DESCRIPTION = 'Utilities for training models for foreign exchange'
GITHUB_URL = 'https://github.com/timesler/forex-utils'

parent_dir = os.path.dirname(os.path.realpath(__file__))

with open(f'{parent_dir}/README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    packages=[
        'forex_utils',
        'forex_utils.datasets',
        'forex_utils.models',
    ],
    package_data={'': ['*.json', '*.yml']},
    provides=['xt_training'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'plotly',
    ],
)
