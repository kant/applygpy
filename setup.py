#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from setuptools import setup, Extension
import numpy as np


setup(name = 'applygpy',
      version = '0.1',
      author = "Max Zwiessele",
      author_email = "ibinbei@gmail.com",
      description = ("Reoccuring applications for applying GPy to datasets"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "https://github.com/mzwiessele/applygpy",
      packages = ["applygpy",
                  ],
      package_dir={'applygpy': 'applygpy'},
      py_modules = ['applygpy.__init__'],
      test_suite = 'applygpy.tests',
      install_requires=['GPy>=0.8', 'matplotlib', 'pandas', ],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
