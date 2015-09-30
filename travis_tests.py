'''
Created on 30 Sep 2015

@author: Max Zwiessele
'''

#!/usr/bin/env python

import matplotlib 
import nose 

matplotlib.use('svg')
nose.main('applygpy', defaultTest='applygpy/tests', argv=['dummyarg0 --with-coverage --cover-xml --cover-xml-file=coverage.xml --cover-erase --cover-package=applygpy'])  

