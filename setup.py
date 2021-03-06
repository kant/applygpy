#!/usr/bin/env python
# -*- coding: utf-8 -*-

#===============================================================================
# Copyright (c) 2015, Max Zwiessele
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of paramax nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

from __future__ import print_function
import os
import sys
from setuptools import setup, Extension
import numpy as np


setup(name = 'applygpy',
      version = '0.1',
      author = "Max Zwiessele",
      author_email = "ibinbei@gmail.com",
      description = ("Reoccuring applications auxiliary packages for applying GPy to datasets"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels",
      url = "https://github.com/mzwiessele/applygpy",
      packages = ["applygpy",
                  'applygpy.bgplvmpanama',
                  "applygpy.tests"
                  ],
      package_dir={'applygpy': 'applygpy'},
      py_modules = ['applygpy.__init__'],
      test_suite = 'applygpy.tests',
      install_requires=['GPy>=0.8', 'pandas', 'scikit-learn'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
