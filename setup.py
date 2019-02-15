# #!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'food_detector'
    ],
    package_dir={'': 'src'},
    install_requires=[
        'scipy>=1.2.1'
    ]
)
setup(**d)
