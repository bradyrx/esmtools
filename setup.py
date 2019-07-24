from setuptools import find_packages, setup

DISTNAME = 'esmtools'
VERSION = '0.1'
AUTHOR = 'Riley X. Brady'
AUTHOR_EMAIL = 'riley.brady@colorado.edu'
DESCRIPTION = (
    'A toolbox for functions related to Earth System Model'
    + 'analysis, with a focus on biogeochemical oceanography.'
)
URL = 'https://github.com/bradyrx/esmtools'
LICENSE = 'MIT'
INSTALL_REQUIRES = [
    'xarray',
    'pandas',
    'numpy',
    'matplotlib',
    'xskillscore',
    'tqdm',
    'climpred >= 1.0.1',
    'cftime',
]
# TODO: Add testing
# TESTS_REQUIRE = ['pytest']
PYTHON_REQUIRE = '>=3.6'

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRE,
)
