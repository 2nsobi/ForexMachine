from setuptools import setup

# List of dependencies installed via `pip install -e .`
# by virtue of the Setuptools `install_requires` value below.
requires = [
    'ta',
    'matplotlib',
    'numpy',
    'pandas',
    'pyyaml',
    'PyQt5'
]

# List of dependencies installed via `pip install -e ".[dev]"`
# by virtue of the Setuptools `extras_require` value in the Python
# dictionary below.
dev_requires = [
    'pytest',
    'notebook',
]

setup(
    name='ForexMachine',
    install_requires=requires,
    extras_require={
        'dev': dev_requires
    },
)