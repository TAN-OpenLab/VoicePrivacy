#!/usr/bin/python
"""
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import imp
import numpy as np
import random
from random import shuffle as random_shuffle
from pyTools import display
from ioTools import readwrite

def lstdirNoExt(fileDir, ext=None):
    """ return the list of file names without extension                  
    """
    #return [x.split('.')[0] for x in os.listdir(fileDir)]
    if ext is None:
        return [x.split('.')[0] for x in os.listdir(fileDir) if not x.startswith('.')]
    else:
        return [x.split('.')[0] for x in os.listdir(fileDir) if not x.startswith('.') and x.endswith(ext)]

def crossSet(list1, list2):
    """ return the cross-set of list1 and list2                          
    """
    return list(set(list1).intersection(list2))


def createFileLst(dataDirs, dataExts, dataDim, dataListDirs, trainortest, trainSetRatio=0.8, random_seed=12345):
    """ create data lists 
        output *.scp will be in dataListDirs
    """
    dataDirs = dataDirs.split(',')
    dataExts = dataExts.split(',')
    dataDims = [int(x) for x in dataDim.split('_')]
    assert len(dataDirs) == len(dataExts), 'Error: sub_1_prepare_list.py dataDirs and dataExts wrong'
    
    # get the cross-set of file lists
    dataList  = lstdirNoExt(dataDirs[0], dataExts[0])
    for dataDir, dataExt in zip(dataDirs[1:], dataExts[1:]):
        listTmp  = lstdirNoExt(dataDir, dataExt)
        dataList = crossSet(dataList, listTmp)

    # check if file exists
    if len(dataList) < 1:
        display.self_print("Error: fail to generate file list. Please check:", 'error')
        display.self_print("path_acous_feats, ext_acous_feats, path_waveform in config.py;", 'error')
        display.self_print("Please also check the names of input data files.", 'error')
        raise Exception("Error: fail to generate file list.")
        
    # randomize the data file list
    # sort at first, 
    dataList.sort() 
    random.seed(random_seed)
    random_shuffle(dataList)

    # before start, take a simple test on the configuration of feature dimension
    frameNum = None
    for inputDir, featDim, featName in zip(dataDirs[0:-1], dataDims[0:-1], dataExts[0:-1]):
        inputFile = os.path.join(inputDir, dataList[0]) + '.' + featName.lstrip('.')
        if os.path.isfile(inputFile):
            tmpframeNum = readwrite.read_raw_mat(inputFile, featDim).shape[0]
            if frameNum is None or frameNum < tmpframeNum:
                frameNum = tmpframeNum
                
    for inputDir, featDim, featName in zip(dataDirs[0:-1], dataDims[0:-1], dataExts[0:-1]):
        inputFile = os.path.join(inputDir, dataList[0]) + '.' + featName.lstrip('.')
        if os.path.isfile(inputFile):
            tmpframeNum = readwrite.read_raw_mat(inputFile, featDim).shape[0]
            if np.abs(frameNum - tmpframeNum)*1.0/frameNum > 0.1:
                if featDim == readwrite.read_raw_mat(inputFile, 1).shape[0]:
                    pass
                else:
                    display.self_print("Large mismatch of frame numbers %s" % (inputFile))
                    display.self_print("Please check whether inputDim are correct", 'error')
                    display.self_print("Or check input features are corrupted", 'error')
                    raise Exception("Error: mismatch of frame numbers")
    
    
    display.self_print('Generating data lists in to %s' % (dataListDirs), 'highlight')

    if trainortest == 'test':
        # generating for test set
        display.self_print('\ttest size: %d' % (len(dataList)), 'highlight')
        testFileOut = dataListDirs + os.path.sep + 'test.lst'
        testFilePtr = open(testFileOut, 'w')
        for fileName in dataList:
            testFilePtr.write('%s\n' % (fileName))
        testFilePtr.close()
    else:
        # determine the train and validatition set if necessary
        if trainSetRatio is not None and trainSetRatio > 0.0:
            if trainSetRatio < 1.0:
                # a ratio
                trainSetDivide = int(np.round(len(dataList)*trainSetRatio))
            elif trainSetRatio < len(dataList):
                # absolute value of the number of training utterances
                trainSetDivide = int(trainSetRatio)
            else:
                # a default ratio 0.8
                display.self_print('Warning: train_utts = 0.8 is used to divide train/val', 'warning')
                trainSetDivide = int(np.round(len(dataList)*0.8))
                
            trainSet = dataList[0:trainSetDivide]
            valSet = dataList[trainSetDivide:]

            if len(valSet) > len(trainSet):
                display.self_print("Warning: validation set is larger than training set", 'warning')
                display.self_print("It's better to change train_utts in config.py", 'warning')
            
            trainFileOut = dataListDirs + os.path.sep + 'train.lst'
            trainFilePtr = open(trainFileOut, 'w')
            for fileName in trainSet:
                trainFilePtr.write('%s\n' % (fileName))
            trainFilePtr.close()

            if len(valSet):
                valFileOut = dataListDirs + os.path.sep + 'val.lst'
                valFilePtr = open(valFileOut, 'w')
                for fileName in valSet:
                    valFilePtr.write('%s\n' % (fileName))
                valFilePtr.close()
                
            display.self_print('\ttrain/val sizes: %d, %d' % (len(trainSet), len(valSet)), 'warning')
        else:
            display.self_print('\ttrain/val sizes: %d, 0' % (len(dataList)), 'warning')
            trainFileOut = dataListDirs + os.path.sep + 'train.lst'
            trainFilePtr = open(trainFileOut, 'w')
            for fileName in dataList:
                trainFilePtr.write('%s\n' % (fileName))
            trainFilePtr.close()
    # done
    return

    

if __name__ == "__main__":
    
    dataDirs = sys.argv[1]
    dataExts = sys.argv[2]
    dataDims = sys.argv[3]
    dataListDir = sys.argv[4]
    try:
        trainRatio = float(sys.argv[5])
    except ValueError:
        trainRatio = None

    try:
        if sys.argv[6] == 'testset':
            trainortest = 'test'
        else:
            trainortest = 'train'
    except IndexError:
        trainortest = 'train'        
        
    try:
        os.mkdir(dataListDir)
    except OSError:
        pass
    createFileLst(dataDirs, dataExts, dataDims, dataListDir, trainortest,
                  trainSetRatio=trainRatio)
        
