import setuptools

PACKAGE_NAME = 'd2d'


setuptools.setup(
    name='d2d',
    version='0.0.1',
    url='https://github.com/barelmas245/BNet21-D2D',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'networkx',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'pathlib',
        'matplotlib',
        'pykml'
    ],
)
