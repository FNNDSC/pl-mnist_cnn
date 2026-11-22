import re

from setuptools import setup

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")


def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.

    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    with open(rel_path, 'r') as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f'Could not find __version__ in {rel_path}')
        return version.group(0)


setup(
    name='pl-mnist_cnn',
    version=get_version('app.py'),
    description='Train, evaluate, or predict with a small CNN on MNIST handwritten digits.',
    author='David Edelist',
    author_email='jedelist@bu.edu',
    url='https://github.com/FNNDSC/pl-mnist_cnn',
    py_modules=['app'],
    packages=['mnist_cnn'],
    install_requires=[
        'chris_plugin==0.4.0',
        'torch>=2.2',
        'torchvision>=0.17',
        'numpy',
        'tqdm>=4.66',
        'Pillow>=10.0',
    ],
    license='MIT',
    entry_points={
        'console_scripts': [
            'mnist_cnn = app:main'
        ]
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    extras_require={
        'none': [],
        'dev': [
            'pytest~=8.0'
        ]
    }
)
