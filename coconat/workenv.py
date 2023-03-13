# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:40:53 2017

@author: cas
"""

import tempfile
import shutil
import os

class TemporaryEnv():
  def __init__(self, basedir):
    tempfile.tempdir = os.path.abspath(tempfile.mkdtemp(prefix="job.tmpd.", dir=basedir))

  def destroy(self):
    if not tempfile.tempdir == None:
      shutil.rmtree(tempfile.tempdir)

  def createFile(self, prefix, suffix):
    outTmpFile = tempfile.NamedTemporaryFile(mode   = 'w',
                                             prefix = prefix,
                                             suffix = suffix,
                                             delete = False)
    outTmpFileName = outTmpFile.name
    outTmpFile.close()
    return outTmpFileName

  def createDir(self, prefix):
    outTmpDir = tempfile.mkdtemp(prefix=prefix)
    return outTmpDir
