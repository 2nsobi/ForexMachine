from setuptools import setup


def read_file(path):
    with open(path, 'r') as fp:
        return fp.read()


# List of dependencies installed via `pip install -e .`
# by virtue of the Setuptools `install_requires` value below.
requires = read_file('./requirements.txt')

# List of dependencies installed via `pip install -e ".[dev]"`
# by virtue of the Setuptools `extras_require` value in the Python
# dictionary below.
dev_requires = read_file('./dev_requirements.txt')

setup(
    name='ForexMachine',
    install_requires=requires,
    extras_require={
        'dev': dev_requires
    },
)
