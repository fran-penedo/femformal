try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Formal Methods for PDEs',
    'url': '',
    'author': 'Fran Penedo',
    'author_email': 'franp@bu.edu',
    'version': '0.1.1',
    'install_requires': [],
    'packages': ['femformal'],
    'scripts': [],
    'name': 'femformal'
}

setup(**config)
