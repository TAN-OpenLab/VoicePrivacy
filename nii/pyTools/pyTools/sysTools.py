#!/usr/bin/python
"""
This script defines some tools for python
"""
from __future__ import absolute_import
from __future__ import print_function
import os, sys

def selfSystem(commandline, justPrint, comment=None):
    if justPrint == False:
        os.system(commandline)
        if comment is not None:
            print("%s %s" % (commandline, comment))
    else:
        if comment is None:
            comment = ''
        print("%s %s" % (commandline, comment))



