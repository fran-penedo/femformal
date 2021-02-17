try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    "description": "Formal Methods for PDEs",
    "url": "",
    "author": "Fran Penedo",
    "author_email": "franp@bu.edu",
    "version": "0.1.2",
    "install_requires": [
        "stlmilp[milp]@git+git://github.com/fran-penedo/templogic.git@v1.0.1",
        "enum34>=1.1.6",
        "matplotlib>=1.5.2",
        "numpy>=1.11.1",
        "scipy>=0.17.1",
        "gurobipy>=9.1.1",
    ],
    "packages": ["femformal"],
    "scripts": [],
    "name": "femformal",
}

setup(**config)
