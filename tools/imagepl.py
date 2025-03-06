# Python 3.7.2 version of the ODELAY Image Pipeline

from fast_histogram import histogram1d
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np


import warnings
import multiprocessing as mp
import os
import json
import pandas as pd
import sqlalchemy as db
import pathlib
import re
import scipy.io as sio
from scipy.sparse   import csr_matrix
from scipy.optimize import minimize 
import time

import cv2 
for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

# internal libraries
import tools.fileio     as fio 
import tools.odelayplot as odp 

def readImage(fileName):

    imageData = fio.loadmatlab(fileName)

    
    # fhandle = h5py.File(fileName, 'r')
    return imageData 

def parseImage(imageFilePath):
    '''
    Parse dictionaries read from both *.mat files and from hdf5 files. So that each file types returns the same dictionary.
    '''

    imageDict = fio.loadData(imageFilePath)

    # Matlab saved files 
    if 'xyzTime' in imageDict.keys():
        imageData = {}
        imageData['Bf'] = {}
        imageData['Bf']['imageStack'] = imageDict['rawImage']
        imageData['Bf']['xyzTime']    = imageDict['xyzTime']
        # imageData['roi'] = imageDict['wellNum']
    
        if 'fluorImage' in [*imageDict]:
            for key in imageDict['fluorImage'].keys():
                imageData[key] = {}
                imageData[key]['imageStack'] =  imageDict['fluorImage'][key]['rawImage']
                imageData[key]['xyzTime']    =  imageDict['xyzTime']   
    else:
        imageData = imageDict
                
    return imageData

def readExcelSheetDisc(fileName):
    
    if fileName == None:
        fileName = fio.openFileDialog()

    df = pd.read_excel('fileName', sheetname='Sheet1')

    print("Column headings:")
    print(df.columns)

def readExpDisc(fileName):
    # Reture ExpObj
    if fileName ==None:
        fileName = fio.openFileDialog()

    expData = fio.loadData(fileName)


    return expData

def loadStageData(imagePath):
    stageFileMatlab = imagePath / 'ODELAY_StageData.mat'
    stageFilePython = imagePath / 'odelayExpConfig.cfg'
    # zStagePosPython = imagePath / 'odelayZstate.hdf5'
  
    if stageFileMatlab.exists():
        stageData = fio.loadData(stageFileMatlab)
        roiIndex = stageData['mP']['wellIdx']-1
        roiList  = list(stageData['mP']['wellID'][roiIndex])
        roiList.sort()
        stageData['roiIndex'] = stageData['mP']['wellIdx']-1
        stageData['roiList']  = roiList
        stageData['mag']      = stageData['mP']['mag']
        stageData['pixSize']      = stageData['mP']['pixSize']


    elif stageFilePython.exists():
        with open(stageFilePython, 'r') as fileIn:
            stageData = json.load(fileIn)
            # zPos = fio.loadData(zStagePosPython)

    return stageData

def initializeExperiment(imagePath, dataPath):
    '''

    Write ODELAY Index File to initialize experiment and provide a list of roi to process as well as experiment variables.
    Critical variables:
    starting time--must be before all file time points
    magnification
    pixel size
    sensor size

    Future versions of the microscope control software will write this data into the images.

    1. Make ROI Dict that includes Paths to files and number of images in each file.
    2. Make Dict of microscope parameters magnification and pixel size and sensor data
    3. Write those variables to a hdf5 file for retrival by workers processing each ROI individually.
   
    '''
    # Parse argument and check to see if it is a path file.

    if isinstance(imagePath, str):
        imagePath = pathlib.Path(imagePath)
    if isinstance(dataPath, str):
        dataPath = pathlib.Path(dataPath)

    expName = imagePath.parts[-1]
    stageData = loadStageData(imagePath)

    # Read in which folders are there and check 
    roiFiles = getRoiFileList(imagePath, stageData['roiList'])

    backgroundImage = generateBackground(imagePath, stageData['roiList'])
   
    odelayDataPath = dataPath / 'ODELAY Roi Data'

    if not odelayDataPath.exists():
        odelayDataPath.mkdir()

    initFileName = expName + '_Index_ODELAYData.hdf5'
    expInitFilePath = dataPath / initFileName

    expDictionary = {
        'backgroundImage': backgroundImage,
        'defaultFitRanges': np.array([0,0]),
        'maxObj': 5000,
        'numTimePoints': 320,   # number of timeponts
        'timerIncrement': 1800, # timer increment in seconds
        'threshold_offset': 1,
        'pixSize': stageData['pixSize'],
        'sensorSize': np.array(backgroundImage.shape,dtype='int32'),
        'magnification': 20,#stageData['mag'],
        'coarseness': 25,
        'kernalerode': 3,
        'kernalopen': 8,
        'roiFiles': roiFiles,
        'experiment_name': expName,
        'odelayDataPath': str(odelayDataPath),
        'expInitFilePath': str(expInitFilePath)
    }
    
    fio.saveDict(expInitFilePath, expDictionary)

    return expDictionary 

def generateBackground(imagePath, roiList):
    '''
    Generate sensor background by averaging a number of initial images given by the length of the roiList.
    '''
    # ToDo: add in multicolor support for fluorescent images 

    numImage = len(roiList)

    roiPath = imagePath / roiList[0]
    imageFilePath = next(roiPath.glob('*.[mh][ad][tf]*')) 
 
    imageData = parseImage(imageFilePath)
 
    imageDim   = imageData['Bf']['imageStack'].shape
    accumeImage = np.zeros(imageDim[0:2], dtype= 'float')

    imageDevisor = float(numImage * imageDim[2])

    for im in range(numImage):
        roiPath = imagePath / roiList[im]
        imageFilePath = next(roiPath.glob('*.[mh][ad][tf]*')) 
      
        imageData = parseImage(imageFilePath)
       
        for tile in range(imageDim[2]):
            floatImage = (1/imageDevisor) * imageData['Bf']['imageStack'][:,:,tile].astype('float')
            accumeImage += floatImage

  
    accumeImage-= np.min(accumeImage)

    return accumeImage.astype('uint16')

def odelay_localProcess(roiList = None, process_count = None):

    # roiList = ['E07', 'E08', 'E09', 'E10', 'E11']
    # process_count = 5
   
    if process_count == None:
        process_count = 6 #mp.cpu_count()-2
	
    if process_count<1:
        process_count = 1

    print(f'process_count = {process_count}, of {mp.cpu_count()}')
    configfile = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )
    with open(configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])

    expData = initializeExperiment(imagePath, dataPath)

    if roiList == None:
        roiList =  [*expData['roiFiles']]

    procs     = []
    proc_coms = []
    worker_status = []

    # roiDict = roiProcess(imagePath, dataPath, 'E07')

    for ind in range(process_count):    
        roi = []

        conn1, conn2 = mp.Pipe([True])
        proc_coms.append(conn1)

        worker_status.append('ready')

        proc = mp.Process(target = odelay_SubProcess, args = (roi, conn2,))
        procs.append(proc)
        proc.start()

    working = True
    flag = False
    while len(roiList) > 0:
        for worker, com in enumerate(proc_coms,0):
            # Poll the communication pipe for ready state and send image file path
            while com.poll() and len(roiList)>0:
                print('worker polled')
                
                comrcv = com.recv()
                print(comrcv[0])
                
                if (comrcv[0] == 'ready'):                    
                    roi = roiList.pop(0)
                    print(f'Starting Process {roi}')
                    com.send([roi])
                    worker_status[worker] = 'working'

    print('ROI list empty')

    while not all(status == 'ready' for status in worker_status):
        for worker, com in enumerate(proc_coms, 0):
            while com.poll():
                comrcv = com.recv()
                worker_status[worker] = comrcv[0]   

    for com in proc_coms:
        try:
            com.send(['kill'])
            com.close()
            print('Sent Kill All')
        except:
            print('Connection Already Closed')


    for proc in procs:
        try:
            proc.kill()
            print('Killing All Processes')
        except:
            print('Proces Dead')

    return None

def odelay_SubProcess(roi, conn):
   
    configfile = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )
    
    with open(configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    
    run = True
    conn.send(['ready'])

    while run: 
        if conn.poll(200): 
            command = conn.recv()

            if command == 'kill':
                conn.close()
                print(command)
                run = False
            else:
                try:
                    roi = command
                    print(f'SubProcess {roi[0]} starting')
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        roiDict = roiProcess(imagePath, dataPath, roi[0])
                    conn.send(['ready'])

                except Exception as e: 
                    print(str(e))
                    print(f'Error in {roi}')
                    conn.send(['ready'])
        else:
            conn.close()
            print('No Jobs in time limit, closed process')
            run = False

    return None

def roiProcess(imagepath, datapath, roiID, verbos = False):

    '''
    Data from Experiment Dictionary or Object
    '''
    if isinstance(imagepath, str):
        imagePath = pathlib.Path(imagepath)
    else:
        imagePath = imagepath

    if isinstance(datapath, str):
        dataPath = pathlib.Path(datapath)
    else:
        dataPath = datapath

    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')


    assembleImageSwitch = False
    imPixList = [k for k in dataPath.glob('*impix.hdf5')] 
    if len(imPixList)==1:
        imPixDict = fio.loadData(imPixList[0])
        assembleImageSwitch = True

    expData = fio.loadData(expIndexPath)

    #####################################
    # Load Dictionary variables  There has to be a way to dynamically add these
    #####################################
    background       = expData['backgroundImage']
    defaultFitRanges = expData['defaultFitRanges']
    maxObj           = expData['maxObj']
    numTimePoints    = expData['numTimePoints']   # number of timeponts
    timerIncrement   = expData['timerIncrement'] # timer increment in seconds
    threshold_offset = expData['threshold_offset']
    pixSize          = expData['pixSize']
    sensorSize       = expData['sensorSize']
    magnification    = expData['magnification']
    coarseness       = expData['coarseness']
    kernalerode      = expData['kernalerode']
    kernalopen       = expData['kernalopen']
    roiFiles         = expData['roiFiles']
    experiment_name  = expData['experiment_name']
    odelayDataPath   = dataPath / 'ODELAY Roi Data'

    ############################
    # expData dictionary is a hdf5 file that will contain the correct information 
    # initialize the experiment.  Perhaps it should be an ini file but at the momement its not
    # defaultFitRanges = None
    # maxObj = 5000
    # numTimePoints = 320   # number of timeponts
    # timerIncrement = 1800 # timer increment in seconds
    # threshold_offset = 1
    # pixSize = 6.45
    # magnification = 20
    # courseness = 25
    # kernalerode = 3
    # kernalopen  = 8
    ############################
   
    # monitorData = fio.loadmat(monitorDataFile)
    # % Load Well Data
    # TODO: loadWell State for cronjob or monitor data files
    #  Load state from Database or create one if it doesn't exist
    #  Check number of images analyzed and number not analyzed 
    #  NewLoadImage +
    #  LoadOldImage +
    #  ThresholdOldImage +
    #  ThresholdNewImage +
    #  PhaseCorrelate Old New Evaluate SampleDrift +
    #  BlobAnalysis +
    #  Object Track -+
    #  EnterData into ObjectNext and ObjectTrack Data -+
    #  Estimate Growth curves -+
    #  Save Low Bit Depth Image for display  
    #  Update well analysis 
    #  Shut down workers once caught up. 
    
    '''
    The following code is to initialize data for all wells
    '''

    if isinstance(roiID, str):
        roiLabel = roiID

    elif isinstance(roiID, int):
        roiList = [*roiFiles]
        roiLabel = roiList[roiID]
    # Else this will crash
    
    roiPath = imagePath /  roiLabel
    imageFileList = os.listdir(roiPath)
    # Understand this gem of a regular expression sort.
    imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
  
    # [imageIndex, imageFileList] = checkFocus(imagePath, expData, roiID)

    numImages = len(imageFileList)
    
    print(f'{roiID} has {numImages} images')

    if numTimePoints<numImages: 
        numTimePoints = numImages

    isMatlab = '.mat' in imageFileList[0]

    threshold = np.zeros(numTimePoints, dtype='uint16') # Array 1 x numTimePoints uint16
    # imageFileList = []# List of strings
    stitchMeta    = {} # Dictionary or list for image stitching data
    xyzTime       = np.zeros((numTimePoints, 4), dtype  ='float64')
    timePoints    = np.full( numTimePoints, 'nan', dtype='float64') # Array dbl  1 x numTimePoints double
    
    objectNext    = np.zeros((maxObj, numTimePoints), dtype='uint16') # Array maxObj x numTimePoints uint16
    objectTrack   = np.zeros((maxObj, numTimePoints), dtype='uint16') # Array maxObj x numTimePoints uint16
    objectArea    = np.zeros((maxObj, numTimePoints), dtype='uint32') # Array maxObj x numTimePoints double
    objectCentX   = np.zeros((maxObj, numTimePoints), dtype='float64') # Array maxObj x numTimePoints double
    objectCentY   = np.zeros((maxObj, numTimePoints), dtype='float64') # Array maxObj x numTimePoints double
    numObj        = np.zeros(numTimePoints, dtype = 'float64')
    sumArea       = np.zeros( numTimePoints, dtype = 'float64')
    fitData       = np.zeros((maxObj, 17),   dtype='float64') # Dictionary array maxObj x 17 double
    imageHist     = np.zeros((numTimePoints, 2**16), dtype = 'uint32')
    analyzeIndex  = np.zeros(numTimePoints,  dtype = 'bool')
    xyDisp        = np.zeros((numTimePoints, 4), dtype  = 'int32')
    prImage ={}
    # End Initialization
    
    # processTime = np.zeros()
    tstart = time.time()
    # print(f'The ROI is {roiID}')
    # Start Processing Data Here
    for aI in range(numImages):
        t0 = time.time()
        # load New Image    

        imageFilePath = roiPath / imageFileList[aI]
        if assembleImageSwitch:
            imPix = imPixDict[roiID]
            anImage = assembleImage(imageFilePath, pixSize, magnification, background, imPix.astype(int))
        else:
            anImage = stitchImage(imageFilePath, pixSize, magnification, background)
        #TODO: Generate a thumbnail of the stitched image for use in the GUI later

        # breakpoint()

        stitchMeta.update({f'{aI:03d}': anImage['stitchMeta']})
   
        xyzTime[aI,:] = anImage['stitchMeta']['xyzTime'][0:4]
        xyDim = anImage['Bf'].shape
        
        sobelBf   = SobelGradient(anImage['Bf'])
        sobelCent = SobelGradient(anImage['centIm'])
        
        threshold[aI] = thresholdImage(sobelBf, threshold_offset, coarseness)
        imageHist[aI,:] = histogram1d(sobelBf.ravel(), 2**16, [0,2**16], weights = None).astype('uint32')

        bwBf = np.greater(sobelBf, threshold[aI]).astype('uint8')

        akernel = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], dtype='uint8')

        # dilate
        # fill
        # erode
        # open 
        # bwBf     = cv2.dilate(bwBf, akernel, iterations = 1)
        # okernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalopen , kernalopen))
        # bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_CLOSE,okernel)
        # bwBf     = cv2.erode( bwBf, akernel, iterations = 1)
        # bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_OPEN, okernel)
        
        #######
        # Python Implementation
        ekernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalerode, kernalerode))
        okernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalopen , kernalopen))
        bwBf     = cv2.dilate(bwBf, ekernel, iterations = 1)
        bwBf     = cv2.erode( bwBf, ekernel, iterations = 1)
        bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_OPEN, okernel)
        bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_CLOSE,okernel)

        bwBf[1, :] = 1
        bwBf[:, 1] = 1
        bwBf[:,-1] = 1
        bwBf[-1,:] = 1
        
        sumArea[aI] = np.sum(bwBf)
        anImage['sobelBf'] = sobelBf
        anImage['bwBf'] = bwBf

        imageStats  = cv2.connectedComponentsWithStats(bwBf, 8, cv2.CV_32S)
        # imageStats[0] is the number of objects detected
        # imageStats[1] is the labeled image uint32
        # imageStats[2] is a number of objects x 5 List that are object stats
        # imageStats[3] is object centroids

        # TODO: Extract Fluorescence data from Fluoresences image
        # This will be done either using the threshold areas in the 
        # labeled Image to extract corresponding areas in the 
        # fluoresence image and then summing those areas
     
        if aI != 0:
         
            # Centroid Association
            # Figure out what the image shift is from the previous Images
           
            bw1 = np.greater(sobelCent, threshold[aI]).astype('uint8')
            bw2 = np.greater(prImage['sobelCent'], threshold[aI]).astype('uint8')
            # Use FFT phase corelation to determin the offet
         
            fT = np.multiply(anImage['fTrans'], prImage['fTrans'].conj())
            fTabs = np.divide(fT,abs(fT))
            fmag1 = np.fft.ifft2(fTabs)
            fmag1[0,0] = 0  # The first index of fmag is always 1 so ignor it.
            r, c = np.where(fmag1 == fmag1.max())
            xyDim = anImage['centIm'].shape
            row = [xyDim[0]-r[0], r[0]]
            col = [xyDim[1]-c[0], c[0]]
            rDisp = np.zeros((16,3), dtype = 'int32')
            cDisp = np.zeros((16,3), dtype = 'int32')
            cnt  = 0
           
            for r in row:
                for c in col:
                    rDisp[cnt,:] = [r,0,r]
                    cDisp[cnt,:] = [c,0,c]
                    cnt += 1
                    rDisp[cnt,:] = [0,r,r]
                    cDisp[cnt,:] = [0,c,c]
                    cnt += 1
                    rDisp[cnt,:] = [r,0,r]
                    cDisp[cnt,:] = [0,c,c]
                    cnt += 1
                    rDisp[cnt,:] = [0,r,r]
                    cDisp[cnt,:] = [c,0,c]
                    cnt += 1
           
            cond = np.zeros(16,dtype = 'int32')
            for n in range(16):
                sw1 = np.zeros((xyDim[0] + rDisp[n,2] , xyDim[1] + cDisp[n,2]), dtype = 'uint8')
                sw2 = np.zeros((xyDim[0] + rDisp[n,2] , xyDim[1] + cDisp[n,2]), dtype = 'uint8')
                swT = np.zeros((xyDim[0] + rDisp[n,2] , xyDim[1] + cDisp[n,2]), dtype = 'uint8')
                rs1 = rDisp[n,0]
                re1 = rDisp[n,0] + xyDim[0] 
                cs1 = cDisp[n,0] 
                ce1 = cDisp[n,0] + xyDim[1]
                
                rs2= rDisp[n,1]
                re2= rDisp[n,1] + xyDim[0] 
                cs2= cDisp[n,1] 
                ce2= cDisp[n,1] + xyDim[1]
                sw1[rs1:re1, cs1:ce1] = bw1
                sw2[rs2:re2, cs2:ce2] = bw2
                swT = sw1*sw2
                cond[n] = swT.sum(axis = None, dtype = 'float')      
            
            ind = cond.argmax()
            
            xyDisp[aI,:] = np.array((rDisp[ind,0],cDisp[ind,0],rDisp[ind,1],cDisp[ind,1]), dtype = 'int32')
            
            # this gives the overlap vector for aligning the images
            # Set image Dimensions so they are identical.
            xyDim = bwBf.shape
            xyDimP = prImage['bwBf'].shape
            maxDim = np.max([xyDim, xyDimP],axis = 0)
            maxDisp = np.array((xyDisp[aI,[0,2]].max(), xyDisp[aI,[1,3]].max()),dtype = 'int32')
            
            # To do include translation from images earlier.
            alDim = np.floor((maxDim-xyDim)/2).astype('int')
            auDim = maxDim-np.ceil((maxDim-xyDim)/2).astype('int')
            
            blDim = np.floor((maxDim-xyDimP)/2).astype('int')
            buDim = maxDim-np.ceil((maxDim-xyDimP)/2).astype('int')
            
            arsV = alDim[0] + xyDisp[aI,0]
            areV = auDim[0] + xyDisp[aI,0]
            acsV = alDim[1] + xyDisp[aI,1]
            aceV = auDim[1] + xyDisp[aI,1]
            brsV = blDim[0] + xyDisp[aI,2]
            breV = buDim[0] + xyDisp[aI,2]
            bcsV = blDim[1] + xyDisp[aI,3]
            bceV = buDim[1] + xyDisp[aI,3]

            A    = np.zeros((maxDim + maxDisp),dtype = 'uint8')
            B    = np.zeros((maxDim + maxDisp),dtype = 'uint8')
            aLbl = np.zeros((maxDim + maxDisp),dtype = 'uint16')
            bLbl = np.zeros((maxDim + maxDisp),dtype = 'uint16')

            A[arsV:areV,acsV:aceV]  =  bwBf
            B[brsV:breV,bcsV:bceV]  =  prImage['bwBf']
            aLbl[arsV:areV,acsV:aceV]  =  imageStats[1]
            bLbl[brsV:breV,bcsV:bceV]  =  prevImStats[1]
            
           
            # % Multiply black and white Images together.  This makes a mask
            # % where colonies overlap.
            M = A*B
            ALbl = aLbl*M # Current Labeled Image
            BLbl = bLbl*M # Prev Labeled Image
            
            ccM = cv2.connectedComponents(M, 8, cv2.CV_32S)
            numObj[aI] = ccM[0]
            if ccM[0] >5000:
                print('Number of objectes in ', aI, ' greater than 5000')
            # ccM is the total number of objects returned in the image
            ARvl = ALbl.ravel()
            BRvl = BLbl.ravel()
            MRvl = ccM[1].ravel()

            # Create a sparce matrix of the labeled connected component image 
            smM = csr_matrix((MRvl, [MRvl, np.arange(MRvl.shape[0])] ),
                              shape=(ccM[0],MRvl.shape[0]))

            # Get the indices of the non-zero elements of the connected 
            # connected components. Use a list comprehension and 
            # np.split to find the indicies of each labled area in the ccM 
            # matrix.  Then make sure that the lables of ALbl and BLbl are
            # unique by taking the absolute value of the difference between
            # all the Labeled pixels and summing them.  If all pixels are 
            # are identical then that function diffsum should return zero. 
            # If both Labels in each image are unique then no merging of 
            # overlaping objects has occured. 
            
            trkInds = np.array(([ 
                                [ARvl[inds[0]], BRvl[inds[0]]] 
                                for inds in np.split(smM.indices, smM.indptr[1:-1]) 
                                if diffsum(ARvl[inds])==0 and diffsum(BRvl[inds])==0
                                ]), dtype = 'int')
           
            # Place objects that were linked in the Object Next list into an easier to
            # address Object Track List.  
           
            if np.max(trkInds)>=5000:
                tempInds = trkInds>4999
                trkInds[tempInds] = 0

            objectNext[trkInds[:,1],aI-1] = trkInds[:,0]
            rc = objectNext.shape

            nextHist = histogram1d(objectNext[:,aI-1],rc[0],[0,rc[0]],weights = None).astype('int')
            discard = np.where(nextHist>1)

            for val in discard[0]:
                inds = np.where(objectNext[:,aI-1]==val)
                objectNext[inds,aI-1] = 0

            curInds = np.arange(maxObj, dtype = 'int')
            curVec  = curInds[objectTrack[:,aI-1]!=0]
            nextVec = objectTrack[curVec,aI-1]

            if nextVec.shape != 0:
                objectTrack[curVec,aI] = objectNext[nextVec,aI-1]

            curVec = curInds[objectTrack[:,aI]!=0]
            objVec = objectTrack[curVec,aI]
            objectArea[ curVec, aI] = imageStats[2][objVec,4]
            objectCentX[curVec, aI] = imageStats[3][objVec,0]
            objectCentY[curVec, aI] = imageStats[3][objVec,1]

            # Generate Timepoints for this Data-Set
            if isMatlab:
                timePoints[aI] = (xyzTime[aI,3]-xyzTime[0,3])*1440 # Matlab stores data in fractional number of days. Convert to minutes number of minutes in a day
            else:
                timePoints[aI] = (xyzTime[aI,3]-xyzTime[0,3])/60e6 # datetime is in microseconds so converting to minutes.

        elif aI == 0:
            
            curVec = np.arange(imageStats[0], dtype = 'int')
            timePoints[aI] = 0
            objectTrack[0:imageStats[0],0] = np.arange(imageStats[0], dtype = 'uint16')
            objectArea[ curVec, aI] = imageStats[2][curVec,4]
            objectCentX[curVec, aI] = imageStats[3][curVec,0]
            objectCentY[curVec, aI] = imageStats[3][curVec,1]
            
            
        
        # set up for next Image by replacing the previous image information
        prImage  = anImage
        prImage['sobelCent'] = sobelCent
        prevImStats = imageStats
        t1 = time.time()
        # print('Image ', roiLabel,'-', aI, ' took ', t1-t0, ' seconds')
        # print((t1-tstart)/60, ' minutes have elapsed')


    # breakpoint()
    # This is a filter to get rid of very big stpes in the objectArea that
    #  may be due to either loss of focus or other imaging problems
    
    log2Area = np.log2(objectArea.astype('float'))
    diffArea = np.diff(log2Area,axis=1,n=1, append=0)
    diffAreaAbs = np.abs( diffArea)
    dbInds = diffAreaAbs>1
    bgSteps = np.cumsum(dbInds,axis=1)==0
    objectArea[~bgSteps]= 0
        
    indVec = np.arange(maxObj)
    numObs = np.sum(objectArea!=0, axis = 1)
    fitVec = indVec[numObs>5]

    for m in fitVec:
        (fitCols, fitData[m,0:16]) =  fitGrowthCurves(timePoints, objectArea[m,:],defaultFitRanges)

    if len(fitVec)==0:
        fitCols = {'No Data Fit':1}
        
    # returnDict = {'anImage':     anImage,
    #               'prImage':     prImage,
    #               'background':  background,
    #               'stitchMeta':  stitchMeta,
    #               'imageHist':   imageHist,
    #               'timePoints':  timePoints, 
    #               'objectArea':  objectArea, 
    #               'objectTrack': objectTrack,
    #               'objectCentX': objectCentX,
    #               'objectCentY': objectCentY,
    #               'objectNext':  objectNext, 
    #               'threshold':   threshold,
    #               'numObj':      numObj,
    #               'sumArea':     sumArea,
    #               'xyDisp':      xyDisp,
    #               'xyzTime':     xyzTime,
    #               'fitData':     fitData,
    #               'roiLabel':    roiLabel
    #               }
    
    returnDict = {'stitchMeta':  stitchMeta,
                  'imageIndex':  imageHist,
                  'imageHist':   imageHist,
                  'timePoints':  timePoints, 
                  'objectArea':  objectArea, 
                  'objectTrack': objectTrack,
                  'objectCentX': objectCentX,
                  'objectCentY': objectCentY,
                  'objectNext':  objectNext, 
                  'threshold':   threshold,
                  'sumArea':     sumArea,
                  'numObj':      numObj,
                  'xyDisp':      xyDisp,
                  'xyzTime':     xyzTime,
                  'fitData':     fitData,
                  'fitDataCols': fitCols,
                  'roiLabel':    roiLabel
                  }

    fio.saveROI(odelayDataPath, returnDict)

    return returnDict

def roiMacInfo(imagepath, datapath, roiID, verbos = False):

    '''
        Data from Experiment Dictionary or Object
    '''
    if isinstance(imagepath, str):
        imagePath = pathlib.Path(imagepath)
    else:
        imagePath = imagepath

    if isinstance(datapath, str):
        dataPath = pathlib.Path(datapath)
    else:
        dataPath = datapath

    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')

    expData = fio.loadData(expIndexPath)
    #####################################
    # Load Dictionary variables  There has to be a way to dynamically add these
    #####################################
    background       = expData['backgroundImage']
    defaultFitRanges = expData['defaultFitRanges']
    maxObj           = expData['maxObj']
    numTimePoints    = expData['numTimePoints']  # number of timeponts
    timerIncrement   = expData['timerIncrement'] # timer increment in seconds
    threshold_offset = expData['threshold_offset']
    pixSize          = expData['pixSize']
    sensorSize       = expData['sensorSize']
    magnification    = expData['magnification']
    coarseness       = expData['coarseness']
    kernalerode      = expData['kernalerode']
    kernalopen       = expData['kernalopen']
    roiFiles         = expData['roiFiles']
    experiment_name  = expData['experiment_name']
    roiSavePath      = dataPath / 'ODELAY Roi Data' / f'{roiID}.hdf5'


    '''
    The following code is to initialize data for all wells
    '''

    roiPath = imagePath /  roiID
    fileList = os.listdir(roiPath)
    imageFileList = [fileName for fileName in fileList if '.mat' in fileName]
    # Understand this gem of a regular expression sort.
    imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    numImages = len(imageFileList)
    if numTimePoints<numImages: 
        numTimePoints = numImages

    imageInfo = {}
    # Start Processing Data Here
    for aI in range(numImages):
    #     # load New Image    

        imageFilePath = roiPath / imageFileList[aI]

        anImage = stitchImage(imageFilePath, pixSize, magnification, background)
        # TODO: Generate a thumbnail of the stitched image for use in the GUI later
        imageInfo[f'{aI:03d}'] = {}
        imageInfo[f'{aI:03d}']['stitchMeta'] = anImage['stitchMeta']
        imageInfo[f'{aI:03d}']['index'] = aI+1
        # for imType in anImage['imageLabels'].keys()

        # flourImageDict = {colList[val] : val for val in range(len(colList))}
        fluorImageList = [Lbl for Lbl in [*anImage['imageLabels']] if not Lbl=='Bf']
    
        flourDict ={fluorImageList[im]: im for im in range(len(fluorImageList))}

        for flourIm in fluorImageList:
            threshold = thresholdImage(anImage[flourIm],   threshold_offset, coarseness)
            flourBw = morphImage(anImage[flourIm],   kernalerode, kernalopen, threshold)
            imageStats  = cv2.connectedComponentsWithStats(flourBw, 8, cv2.CV_32S)

            FRvl = anImage[flourIm].ravel()
            MRvl = imageStats[1].ravel()

            # Create a sparce matrix of the labeled connected component image 
            smM = csr_matrix((MRvl, [MRvl, np.arange(MRvl.shape[0])]),
                                shape=(imageStats[0],MRvl.shape[0]))
            objIntensity = np.array(([ 
                                        np.sum(FRvl[inds])
                                        for inds in np.split(smM.indices, smM.indptr[1:-1]) 
                                    ]), dtype = 'uint32')

            imageInfo[f'{aI:03d}'][flourIm] = {}
            imageInfo[f'{aI:03d}'][flourIm]['threshold']    = threshold
            imageInfo[f'{aI:03d}'][flourIm]['boundBox']     = imageStats[2]
            imageInfo[f'{aI:03d}'][flourIm]['centroids']    = imageStats[3]
            imageInfo[f'{aI:03d}'][flourIm]['objIntensity'] = objIntensity


    fio.saveDict(roiSavePath, imageInfo)

    return imageInfo

def roiMacSeg(imagepath, datapath, roiID, anchorColor = 'DAPI', imageSize = 160, verbos = False):

    '''
        Data from Experiment Dictionary or Object
    '''
    if isinstance(imagepath, str):
        imagePath = pathlib.Path(imagepath)
    else:
        imagePath = imagepath

    if isinstance(datapath, str):
        dataPath = pathlib.Path(datapath)
    else:
        dataPath = datapath

    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')

    deadDirPath =  dataPath / 'DeadCells'
    if not deadDirPath.exists():
        deadDirPath.mkdir()

    liveDirPath =  dataPath / 'LiveCells'
    if not liveDirPath.exists():
        liveDirPath.mkdir()

    expData = fio.loadData(expIndexPath)
    #####################################
    # Load Dictionary variables  There has to be a way to dynamically add these
    #####################################
    background       = expData['backgroundImage']
    defaultFitRanges = expData['defaultFitRanges']
    maxObj           = expData['maxObj']
    numTimePoints    = expData['numTimePoints']  # number of timeponts
    timerIncrement   = expData['timerIncrement'] # timer increment in seconds
    threshold_offset = expData['threshold_offset']
    pixSize          = expData['pixSize']
    sensorSize       = expData['sensorSize']
    magnification    = expData['magnification']
    coarseness       = expData['coarseness']
    kernalerode      = expData['kernalerode']
    kernalopen       = expData['kernalopen']
    roiFiles         = expData['roiFiles']
    experiment_name  = expData['experiment_name']
    roiSavePath      = dataPath / 'ODELAY Roi Data' / f'{roiID}.hdf5'

    if isinstance(roiID, str):
        roiLabel = roiID

    elif isinstance(roiID, int):
        roiList = [*roiFiles]
        roiLabel = roiList[roiID]
    # Else this will crash
    
    roiPath = imagePath /  roiLabel
    imageFileList = os.listdir(roiPath)
    # Understand this gem of a regular expression sort.
    imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    numImages = len(imageFileList)
    if numTimePoints<numImages: 
        numTimePoints = numImages

    threshold = np.zeros(numTimePoints, dtype='uint16') # Array 1 x numTimePoints uint16
    # imageFileList = []# List of strings
    stitchMeta    = {} # Dictionary or list for image stitching data
    xyzTime       = np.zeros((numTimePoints, 4), dtype  ='float64')
    timePoints    = np.full( numTimePoints, 'nan', dtype='float64') # Array dbl  1 x numTimePoints double
    
    numObj        = np.zeros(numTimePoints, dtype = 'float64')
    sumArea       = np.zeros( numTimePoints, dtype = 'float64')
    fitData       = np.zeros((maxObj, 17),   dtype='float64') # Dictionary array maxObj x 17 double
    imageHist     = np.zeros((numTimePoints, 2**16), dtype = 'uint32')
    analyzeIndex  = np.zeros(numTimePoints,  dtype = 'bool')
    xyDisp        = np.zeros((numTimePoints, 4), dtype  = 'int32')
    prImage ={}

    '''
    The following code is to initialize data for all wells
    '''

    roiPath = imagePath /  roiID
    fileList = os.listdir(roiPath)
    imageFileList = [fileName for fileName in fileList if '.mat' in fileName]
    # Understand this gem of a regular expression sort.
    imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    numImages = len(imageFileList)
    if numTimePoints>numImages: 
        numTimePoints = numImages

    imageInfo = {}
    liveCnt = 0
    deadCnt = 0

    tstart = time.time()
    print(f'The ROI is {roiID}')
    # Start Processing Data Here
    for aI in range(numTimePoints):
    #     # load New Image    
        t0 = time.time()
        imageFilePath = roiPath / imageFileList[aI]

        anImage = stitchImage(imageFilePath, pixSize, magnification, background)
        imShape = anImage['Bf'].shape
        # TODO: Generate a thumbnail of the stitched image for use in the GUI la        imageInfo[f'{aI:03d}'] =        imageInfo[f'{aI:03d}']['stitchMeta'] = anImage['stitchMet        imageInfo[f'{aI:03d}']['index'] = a      
        sobelBf   = SobelGradient(anImage['Bf'])
      
        threshold       = thresholdImage(sobelBf, 1.2, coarseness)
        imageHist[aI,:] = histogram1d(sobelBf.ravel(), 2**16, [0,2**16], weights = None).astype('uint32')
      
        bwBf1 = np.greater(sobelBf, threshold).astype('uint8')

        akernel = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]], dtype='uint8')

        #######
        # Python Implementation
        kernalerode = 4
        kernalopen  = 3

        ekernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalerode, kernalerode))
        okernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalopen , kernalopen))

        bwBf2     = cv2.dilate(bwBf1, ekernel, iterations = 2)
        bwBf3     = cv2.erode( bwBf2, ekernel, iterations = 2)
        
        bwBf3[1, :] = 1
        bwBf3[:, 1] = 1
        bwBf3[:,-1] = 1
        bwBf3[-1,:] = 1
        
        sumArea[aI] = np.sum(bwBf3)
        anImage['bwBf'] = bwBf2  

        bfImageStats  = cv2.connectedComponentsWithStats(bwBf2, 8, cv2.CV_32S)
        imageInfo[f'{aI:03d}'] = {}
        imageInfo[f'{aI:03d}']['Bf'] = {}
        imageInfo[f'{aI:03d}']['Bf']['threshold']    = threshold
        imageInfo[f'{aI:03d}']['Bf']['boundBox']     = bfImageStats[2] # upper left xy lower right xy
        imageInfo[f'{aI:03d}']['Bf']['centroids']    = bfImageStats[3]
        imageInfo[f'{aI:03d}']['Bf']['fixedBB']      = fixedBB(bfImageStats[3],imageSize, imShape)

        fluorImageList = [Lbl for Lbl in [*anImage['imageLabels']] if not Lbl=='Bf']
    
        # flourDict ={fluorImageList[im]: im for im in range(len(fluorImageList))}
        
        for flourIm in fluorImageList:
            threshold = thresholdImage(anImage[flourIm],   1.3, coarseness)
            flourBw   = morphImage(anImage[flourIm],   kernalerode, kernalopen, threshold)
            flImageStats  = cv2.connectedComponentsWithStats(flourBw, 8, cv2.CV_32S)

            FRvl = anImage[flourIm].ravel()
            MRvl = flImageStats[1].ravel()

            # Create a sparce matrix of the labeled connected component image 
            fluorPix = csr_matrix((MRvl, [MRvl, np.arange(MRvl.shape[0])]),
                                        shape=(flImageStats[0],MRvl.shape[0]))
            objIntensity = np.array(([ 
                                        np.sum(FRvl[inds])
                                        for inds in np.split(fluorPix.indices, fluorPix.indptr[1:-1]) 
                                    ]), dtype = 'uint32')

            imageInfo[f'{aI:03d}'][flourIm] = {}
            imageInfo[f'{aI:03d}'][flourIm]['threshold']    = threshold
            imageInfo[f'{aI:03d}'][flourIm]['boundBox']     = flImageStats[2]
            imageInfo[f'{aI:03d}'][flourIm]['centroids']    = flImageStats[3]
            imageInfo[f'{aI:03d}'][flourIm]['objIntensity'] = objIntensity
            imageInfo[f'{aI:03d}'][flourIm]['fixedBB']      = fixedBB(flImageStats[3],imageSize, imShape)
    
        cellBounds = imageInfo[f'{aI:03d}'][anchorColor]['fixedBB']

        for imageColor in anImage['imageLabels']:
            # figure out if image has fluorescent centroid in image.
            imageCents = imageInfo[f'{aI:03d}'][imageColor]['centroids']
        
            # This creates a centIn which is number of fluorescent centroids (rows) x the number of individual cells found (cols)
            # The functions centIn and bound in will check to see if fluorescent centroids are within the cell bounds 
            # and then check to see if either a centroid or a bounding box contain the centroids.

            centIn = np.zeros((imageCents.shape[0], cellBounds.shape[0]), dtype = 'bool')
            boundIn= np.zeros((imageCents.shape[0], cellBounds.shape[0]), dtype = 'bool')
            
            for row in range(imageCents.shape[0]):
                    centIn[row,:] = checkCentroid(imageCents[row,:], cellBounds, 40, 500)

            for col in range(cellBounds.shape[0]):
                    boundIn[:,col] = checkBoundBox(imageCents, cellBounds[col,:], 40, 500)

            imageInfo[f'{aI:03d}'][imageColor]['centIn']  = np.sum(centIn,  axis = 0)
            imageInfo[f'{aI:03d}'][imageColor]['boundIn'] = np.sum(boundIn, axis = 0)
  
        # breakpoint()
        bfCents   = imageInfo[f'{aI:03d}']['Bf']['centIn']
        dapiCents = imageInfo[f'{aI:03d}']['DAPI']['centIn']
        cy5Cents  = imageInfo[f'{aI:03d}']['Cy5']['centIn']
        
        singleBf   = bfCents ==1
        singleDapi = dapiCents == 1
        singleCy5  = cy5Cents  == 1

        deadCell = singleDapi &  singleCy5 #& singleBf
        liveCell = singleDapi & ~singleCy5 #& singleBf

        deadInds  = np.where(deadCell==True)
        liveInds  = np.where(liveCell==True)
       
        if type(deadInds[0]) is not tuple and type(liveInds[0]) is not tuple:
            deadCellInds = np.unique(deadInds[0])
            liveCellInds = np.unique(liveInds[0])
            imageInfo[f'{aI:03d}']['deadCellInds'] = deadCellInds
            imageInfo[f'{aI:03d}']['liveCellInds'] = liveCellInds

            liveBounds = cellBounds[liveCellInds,:]
            deadBounds = cellBounds[deadCellInds,:]

            imageInfo[f'{aI:03d}']['liveBounds'] = cellBounds[liveCellInds,:]
            imageInfo[f'{aI:03d}']['deadBounds'] = cellBounds[deadCellInds,:]


            deadCnt += deadInds[0].shape[0]
            liveCnt += liveInds[0].shape[0]

        print(f'There are {np.sum(deadCell)} dead cells')
        print(f'There are {np.sum(liveCell)} live cells')

        t1 = time.time()
        print('Image ',  aI, ' took ', t1-t0, ' seconds')
        print((t1-tstart)/60, ' minutes have elapsed')
           
    fio.saveDict(roiSavePath, imageInfo)

    return imageInfo

def checkCentroid(cent, bB, minDim, maxDim):
    # check if centroid is within all bounding boxes.  
    # Retruns logical index of which bounding box the centroid is in.  
    x1 = bB[:,0]
    y1 = bB[:,1]
    x2 = bB[:,0]+bB[:,2]
    y2 = bB[:,1]+bB[:,3]
    test1 = x1<=cent[0]
    test2 = x2>=cent[0]
    test3 = y1<=cent[1] 
    test4 = y2>=cent[1]
    test5 = bB[:,2]>=minDim 
    test6 = bB[:,3]>=minDim
    test7 = bB[:,2]<=maxDim 
    test8 = bB[:,3]<=maxDim

    return test1 & test2 & test3 & test4  & test5 & test6 & test7 & test8

def checkBoundBox(cent, bB, minDim, maxDim):
    # check if centroid is within all bounding boxes.  
    # Retruns logical index of which bounding box the centroid is in.  
    x1 = bB[0]
    y1 = bB[1]
    x2 = bB[0]+bB[2]
    y2 = bB[1]+bB[3]
    test1 = x1<=cent[:,0]
    test2 = x2>=cent[:,0]
    test3 = y1<=cent[:,1] 
    test4 = y2>=cent[:,1]
    test5 = bB[2]>=minDim 
    test6 = bB[3]>=minDim
    test7 = bB[2]<=maxDim 
    test8 = bB[3]<=maxDim

    return test1 & test2 & test3 & test4  & test5 & test6 & test7 & test8

def fixedBB(cents, boxSize, imageSize):

    fixedBoundingBox = np.zeros((cents.shape[0], 4), dtype = 'int')

    fixedBoundingBox[:,0] = cents[:,0] - boxSize/2
    fixedBoundingBox[:,1] = cents[:,1] - boxSize/2
    fixedBoundingBox[:,2] = boxSize
    fixedBoundingBox[:,3] = boxSize

    negInds = (fixedBoundingBox[:,0]<0) | (fixedBoundingBox[:,1]<0)
    toBig   = (np.sum(fixedBoundingBox[:,[0,2]], axis = 1)>imageSize[1]) | (np.sum(fixedBoundingBox[:,[1,3]], axis = 1)>imageSize[0])

    correctInds = negInds | toBig
    fixedBoundingBox[correctInds,0] = 0
    fixedBoundingBox[correctInds,1] = 0
    fixedBoundingBox[correctInds,2] = imageSize[0]
    fixedBoundingBox[correctInds,3] = imageSize[1]

    return fixedBoundingBox

def refitGCs(imagepath, datapath, roiID):


    return None

def gompMinBDt(x, tdata, idata):
    '''
    
    '''
    Klag = np.log((3+5**0.5)/2)
    a    = x[0]
    b    = x[1]
    tlag = x[2]
    dT   = x[3]

    yn=a + b*np.exp(-np.exp((Klag/dT)*(dT+tlag-tdata)))
    vals = np.nansum((yn-idata)**2)

    return vals

def gompBDt(x, tdata):
    '''

    '''
    Klag = np.log((3+5**0.5)/2)
    a    = x[0]
    b    = x[1]
    tlag = x[2]
    dT   = x[3]

    vals=a + b*np.exp(-np.exp((Klag/dT)*(dT+tlag-tdata)))

    return vals

def findPrmsGompBDt(vecB,  vecTlag, vecDT, tdata, adata):
    '''
    Corse-grid search for parameters of the Parameterized Gompertz function

    -------
    Input Parameters
    vecB:     array of B paramters to search
    vecTlag:  array of lag times to search
    vecDT:    array of DT times to search
    tData:    ordered array of timepoints
    aData:    corresponding area data 

    Returns array of estamate parameters
    estVec[0] = a estimate
    estVec[1] = B estimate
    estVec[2] = lag time estimate
    estVec[3] = dT or time between max velocity and lag time

    '''
    
    flag=False
    estVec = np.zeros(4, dtype = 'float')
    estVec[0] = np.nanmean(adata[0:5])
    
    K = np.log((3+5**0.5)/2)
    tVec = np.arange(vecTlag.shape[0])
    
    for B in vecB:
        for tp in tVec[:-1]:
            tlag = vecTlag[tp]
            vecDt = vecTlag[tp+1:]-vecTlag[tp]
             
            for dT in vecDt:
                yn=estVec[0]+B*np.exp(-np.exp((K/dT)*(dT+tlag-tdata)))
                ifunc = np.sum((adata-yn)**2)
                if (not flag) or (flag and ifunc < fmin):
                    
                    fmin    = ifunc
                    estVec[1]   = B
                    estVec[2]   = tlag
                    estVec[3]   = dT
                    flag = True
    
    return estVec

def fitGrowthCurves(timeVec, areaData, defaultRanges):

    numTPs = np.sum(areaData!=0)

    aData   = np.log2(areaData[:numTPs])
    tData   = timeVec[:numTPs]
    Nsteps  = 40
    areaMax = np.max(aData)
    factor  = 1.05

    cumsum = np.cumsum(np.insert(aData, 0, 0))
    smthArea =  (cumsum[5:] - cumsum[:-5]) / 5
    x = tData[:-4]
    y = smthArea
    m = np.diff(y[[0,-1]])/np.diff(x[[0,-1]])
    yVals = m*x + y[0]-m*x[0]

    diffVals = smthArea-yVals
    cumVals  = np.cumsum(diffVals)
    
    lagInd  = np.argmin(diffVals)
    texInd  = np.argmax(diffVals)
    vmxInd  = np.argmin(cumVals)
    
    numPos = np.sum(cumVals[vmxInd:]>0)
    estVec = np.zeros(4, dtype = 'float')
    meanArea = np.mean(aData[0:5])
    stdArea  = np.std(aData[0:5])
    estVec[0] = meanArea

    if  lagInd < vmxInd and vmxInd < texInd:
        estVec[2] = tData[lagInd]
        estVec[3] = tData[vmxInd] - tData[lagInd]
        estVec[1] = aData[vmxInd] - meanArea

    elif lagInd < vmxInd and (texInd<lagInd or texInd<vmxInd):
        estVec[2] = tData[lagInd]
        estVec[1] = aData[vmxInd] - meanArea
        estVec[3] = tData[vmxInd] - tData[lagInd]

    elif lagInd < texInd and (vmxInd < lagInd or vmxInd < texInd):
        estVec[2] = tData[lagInd]
        estVec[1] = aData[texInd]  - meanArea
        estVec[3] = (tData[texInd] - tData[lagInd])/2

    else:
        #  Use course grid optimization function findPrmsGompF to find 
        #  a local minima based on the 
    
        vecDT = np.linspace(1,2*tData[-1],Nsteps)
        bmin = 0
        bmax = 16
        vecTlag = np.linspace(1,tData[-1],Nsteps)
        vecB = np.linspace(bmin,bmax,Nsteps)
        estVec= findPrmsGompBDt(vecB,  vecTlag, vecDT, tData, aData)


    Klag = np.log((3+5**0.5)/2)
    
    aLow = meanArea-3*stdArea
    aUp  = meanArea+3*stdArea
    dTLow = 1
    dTUp  = np.max(tData)
    bLow  = 0.001
    bUp   = 16
    lagLow = 0
    lagUp = np.max(tData)
    
    bnds =  [(aLow, aUp), (bLow,bUp), (lagLow, lagUp), (dTLow, dTUp)]

    minFit = minimize(gompMinBDt, estVec, args = (tData, aData), bounds = bnds)

    a    =  minFit.x[0]
    b    =  minFit.x[1]
    Tlag =  minFit.x[2]
    dT   =  minFit.x[3]
    
    Klag = np.log((3+5**0.5)/2)
    Kex  = np.log((3-5**0.5)/2)
    c = Klag/dT
    d = Tlag*c+Klag
    Tex = 2*dT
    TVmax = d/c
    Tplat = (d-Kex)/c
    Vmax = b*c*np.exp(-1)
    if Vmax !=0:
        Td=1/Vmax
    else:
        Td = 0

    if(TVmax>tData[-1]):
        TdFlag=1
    else:
        TdFlag=0
   
    
    if(Tex>tData[-1]):
        TexFlag = 1
    else:
        TexFlag = 0
   
    ATex     = gompBDt(minFit.x, Tplat)
    Aplateau = gompBDt(minFit.x,1e50)
    fitData =np.array([a, b, Tlag, dT, minFit.fun, Tlag, Td, Tex, ATex, Aplateau, TdFlag, TexFlag, TVmax, Tplat, numTPs,  minFit.fun/numTPs], dtype = 'float')
    colList = ['a', 'b','lag', 'dT', 'ssq', 'Tlag', 'Td', 'Tex', 'ATex', 'Aplateau', 'TdFlag', 'TexFlag', 'TVmax', 'Tplat', 'Num Obs', 'ssq per numTimepoints']
    
    fitDataCols = {}
    n = 0
    for key in colList:
        fitDataCols[key] = n
        n+=1


    return fitDataCols, fitData

def checkFocus(imagePath, expData, roiID):

    
    stageDataPath = imagePath / 'odelayZstate.hdf5'
    
    dataLoaded = False
    cntr = 0
    while dataLoaded == False:
        try:
            stageData = fio.loadData(stageDataPath)
            dataLoaded = True
        except:
            time.sleep(1)
            cntr +=1
            if cntr>15:
                dataLoaded == True

    roiList = [roi for roi in expData['roiFiles'].keys()]
    
    roiInd = roiList.index(roiID)

    imageDict = expData['roiFiles'][roiID]
    imageFileList = sorted(imageDict.keys(), key=lambda x: x[1], reverse=False)
    numImages = len(imageFileList)

    imageIndex = np.arange(numImages)
    zDiff   = np.zeros((numImages,), dtype = 'float')
    badInds = np.zeros((numImages,), dtype = 'bool')

    zFocusPos = stageData['zFocusPos'][roiInd,:numImages]

    zDiff[1:] = np.abs(np.diff(zFocusPos))
    outofFocus = zDiff > 50
    badInds[:-1] = outofFocus[:-1] & outofFocus[1:]

    badImages = imageIndex[badInds]

    # imageIndex = np.delete(imageIndex, badImages)

    # for n in badImages:
    #     imageFileList.pop(n)

    return imageIndex, imageFileList

def stitchImage(imageFileName, pixSize, magnification, background):
    magError = 4
    angError = 0.01
    # defualtVecs = np.array([[-54, 974],[-1,42]],dtype = 'float')
    # if pixSize==None:
    #     pixSize = 6.45
    # if magnification==None:
    #     magnification = 20
    '''
    The goal is to find the displacement angle for all the other images.
    minDist array 
    0,1,2
    3,4,5
    6,7,8

    minDist[:, 0] is the starting image
    minDist[:, 1] is the ending image of that comparison so 0,1 are compairing displacements to zero and one
    minDist[:, 2] is the displacement in pixels between the images
    minDist[:, 3 and 4] are x and y (col, row) displacements 
    minDist[:, 5] is the angle in radians of the displacement vector
    minDist[:, 6 and 7] are the locations in (row, col) of the fftxfft max
    minDist[:, 8 and 9] are the displacement vectors for each image

    
    '''
    imageData = parseImage(imageFileName)
    anImage = {}
    # Get stage position values for each image in the group

    xyzTime = imageData['Bf']['xyzTime']
    imDim   = imageData['Bf']['imageStack'].shape
    numTiles = imDim[2]
    rawImage = np.zeros((imDim), dtype = 'uint16')
    fTN      = np.zeros((imDim), dtype = 'complex')

    # Correct Stage positions and set them to a minimum coordiante axis.  
    minXYZ = xyzTime[0,0:2]
    relXY  = np.array((xyzTime[:,1]-minXYZ[1],xyzTime[:,0]-minXYZ[0])).T
    
    pixXYrel = abs(relXY)*magnification/pixSize
    overlapXY = np.full((numTiles,numTiles),0,dtype = 'int')

    # Determine relative displacement between images
    distRows = int(numTiles*(numTiles-1)/2) # Number of comparisons needed 
    distXY   = np.full((distRows,10),'nan',dtype = 'float')
    tempInds = np.full((distRows),False,dtype = bool)

    # calculate distances between image locations
    cnt = 0
    for col in range(numTiles-1):
        for row in range(col+1,numTiles):
            vecXY = pixXYrel[row,:] - pixXYrel[col,:]
            magXY = sum(vecXY**2)**0.5
            distXY[cnt,2] = magXY
            distXY[cnt,0:2]= [col,row]
            distXY[cnt,3:5] = vecXY
            distXY[cnt,5] = np.arctan2(vecXY[0],vecXY[1])

            if ((np.arctan2(0,1)-np.abs(distXY[cnt,5]))<0.01) & (magXY<imDim[1]):
                tempInds[cnt] = True
                overlapXY[row,col] =  1
                overlapXY[col,row] =  3         
        
            elif ((np.arctan2(1,0)-np.abs(distXY[cnt,5]))<0.01) & (magXY<imDim[0]):
                tempInds[cnt] = True
                overlapXY[row,col] =  2
                overlapXY[col,row] =  4
            # Add non-rectungular conditions here.
            cnt = cnt+1

    # overlapXY should look like array([[0, 3, 0, 4, 0, 0, 0, 0, 0],
    #        [1, 0, 3, 0, 4, 0, 0, 0, 0],
    #        [0, 1, 0, 0, 0, 4, 0, 0, 0],
    #        [2, 0, 0, 0, 3, 0, 4, 0, 0],
    #        [0, 2, 0, 1, 0, 3, 0, 4, 0],
    #        [0, 0, 2, 0, 1, 0, 0, 0, 4],
    #        [0, 0, 0, 2, 0, 0, 0, 3, 0],
    #        [0, 0, 0, 0, 2, 0, 1, 0, 3],
    #        [0, 0, 0, 0, 0, 2, 0, 1, 0]])

    minDist = distXY[tempInds,:]
    numComp = sum(tempInds)

    #   TODO figure out how to use overlapXY and image order to determine image
    #   comparison order imCompOrd

    # Load Background Image if it exists
    if not hasattr(background, "shape"):
        background = np.full((imDim[0:2]), 0, dtype = 'uint16')
    # breakpoint()
    #Read in images into RawStack and correct for background
    # for imNum in range(numTiles):
    #    imageData['Bf']['imageStack'][:,:,imNum]-=background
    

    anImage['centIm'] = imageData['Bf']['imageStack'][:,:,5].squeeze()

    for n in range(numTiles):
        fTN[:,:,n] = np.fft.fft2(imageData['Bf']['imageStack'][:,:,n]) # perform fft2 for all images in the stack in prep for alignment.
     
    anImage['fTrans'] = fTN[:,:,5] # Save this transform for later alignment of images
   
    fT    = np.zeros((imDim[0:2]), dtype = 'complex128')
    fTabs = np.zeros((imDim[0:2]), dtype = 'complex128')
    fmag1 = np.zeros((imDim[0:2]), dtype = 'double')
    
    for n in range(numComp):
        # get the FFT of the two images we wish to compare as found by those that overlap.
        # calculate the cross-correlation of the images
        fT = np.multiply(fTN[:,:,int(minDist[n,0])], fTN[:,:,int(minDist[n,1])].conj())
        fTabs = np.divide(fT,abs(fT))
        fmag1 = np.fft.ifft2(fTabs)
        filtMag = cv2.filter2D(fmag1.real, cv2.CV_64F, np.ones((3,3), dtype = 'float'))

        filtMag[0:2,:] = 0 # Supress singlas from the corners as they tend to overwell this routine.  
        filtMag[:,-2:] = 0
        filtMag[:,0:2] = 0
        filtMag[-2:,:] = 0
       

        minDist[n,6], minDist[n,7] = np.unravel_index(np.argmax(filtMag), filtMag.shape) 
       
    #  calculate the displacement vector diffPhaseXY which is the XY
    #  displacement from the stage corrdinates.  The smallest displacement
    #  in this case is the correction since larger displacements are
    #  probably incorrect.
    
    magD        = np.zeros((numComp),  dtype = 'float')
    magI        = np.zeros((numComp),  dtype = 'float')
    magT        = np.zeros((numComp,4),dtype = 'float')
    diffMag     = np.zeros((numComp,4),dtype = 'float')
    magDT       = np.zeros((numComp,4),dtype = 'float')
    angDT       = np.zeros((numComp,4),dtype = 'float')
    magMin      = np.zeros((numComp),  dtype = 'float')
    angMin      = np.zeros((numComp),  dtype = 'float')
    angCheck    = np.zeros((numComp),  dtype = 'int')
    magCheck    = np.zeros((numComp),  dtype = 'int')
    crsCheck    = np.zeros((numComp),  dtype = 'float')
        
    TDvec       = np.zeros((numComp,2),dtype = 'float')

    #  Constrain by calculate upper right hand corner for each image and see which is smallest    
    quadDisp = np.array([0,imDim[1]-1, imDim[0]-1,imDim[1]-1, imDim[0]-1,0,0,0],dtype = int).reshape(4,2)

    for r in range(numComp):
        D = minDist[r,3:5]
        magD[r] = sum(D**2)**0.5
        for c in range(4):
            T = minDist[r,6:8]-quadDisp[c,:]
            magT[r,c] = sum(T**2)**0.5
            magDT[r,c] =  np.abs(magT[r,c]-magD[r])
            angDT[r,c] = np.arccos(np.dot(T,D)/(magD[r]*magT[r,c]))
            
        magCheck[r] = magDT[r,:].argmin()
        magMin[r]   = magDT[r,:].min()
        
        angCheck[r] = angDT[r,:].argmin()
        angMin[r]   = angDT[r,:].min()
        T = minDist[r,6:8] - quadDisp[magCheck[r],:]
        TDvec[r,:] = T-D
        crsCheck[r] = np.cross(T,D) # take cross product of vectors to determin if the vectors are clock or counter clockwise
        
        minDist[r,8:10] = TDvec[r,:]
    # round the angles between the vectors so that the numbers that are
    # close can be calculated.
    # MinDist Key col 0: From Image to Imoge (Key col 1) 
    breakpoint()
    if np.sum(magMin<4)>=5:
        pos_neg  = np.sum(crsCheck[magMin<4])/np.abs(np.sum(crsCheck[magMin<4]))
        theta = pos_neg * np.mean(angMin[magMin<4])
        rot_matrix = np.array([[np.cos(theta), np.sin (theta)],
                               [-np.sin(theta), np.cos(theta)]])
        xy_n = rot_matrix @ pixXYrel.T
        phaseXY = xy_n.T

     


    else:
        # Find the Regions with the same displacement vectors
        sameVec = np.zeros((numComp),dtype = 'int')
        for m in range(numComp):
            sameVec[m] = overlapXY[int(minDist[m,1]),int(minDist[m,0])]
        uniqVec = np.unique(sameVec)

        # round the angles between the vectors so that the numbers that are
        # close can be calculated.
        angFlag  = np.zeros(numComp, dtype = 'int')
        magFlag  = np.zeros(numComp, dtype = 'int')
        angProbs = np.zeros(numComp, dtype = 'bool')
        magProbs = np.zeros(numComp, dtype = 'bool')


        for m in range(numComp): 
            angFlag[m] =  sum((abs(angMin[sameVec==sameVec[m]]-angMin[m])>angError).astype('uint8')) # put Exp variables here
            magFlag[m] =  sum((abs(magMin[sameVec==sameVec[m]]-magMin[m])>magError).astype('uint8')) # put exp variables here

        # This means there is a bad vector as all should be identical
        for m in uniqVec:
            magProbs[sameVec == m ] = magFlag[sameVec == m ] != min(magFlag[sameVec == m ])
            angProbs[sameVec == m ] = angFlag[sameVec == m ] != min(angFlag[sameVec == m ])
        
        numProbs = sum(magProbs | angProbs)
        
        if numProbs > 0:
            vecList = np.arange(numComp)
            fixList = vecList[magProbs|angProbs]
            
            for m in fixList:
                sameInd = sameVec == sameVec[m]
                sameInd[fixList] = False
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category = RuntimeWarning)
                    TDvec[m,:] = np.nanmean(TDvec[sameInd,:],0)

            if sum(np.isnan(TDvec).ravel().astype('uint8'))>0:
                TDvec = np.zeros((numComp,2),dtype = 'float')
        

        # Find vectors paths to each image;
        imagePath = np.zeros((numTiles,numTiles),dtype = 'int')
        numSteps =  np.zeros(numTiles,dtype = 'int')
        minIndex = np.arange(numComp)

        for imNum in range(1,numTiles):
            revOrder = np.zeros(numTiles, dtype = 'int')
            prevImage = imNum
            cnt = 0
            while (prevImage!=0) & (cnt < numTiles):
                val = minIndex[minDist[:,1]==prevImage]
                prevImage = minDist[val[0],0]
                revOrder[cnt] = val[0]
                cnt = cnt+1
                # this is the reverse order of the path and will be flipped
        
            imagePath[imNum,0:cnt] = np.flip(revOrder[0:cnt])
            numSteps[imNum] = cnt
    
        # correct the phaseXY from displacements in the individual images to
        # the cumulative displacement from correcting each image
        phaseXY    = np.zeros((numTiles,2),dtype='float')    
        cumPhaseXY = np.zeros((numTiles,2),dtype='float')

        for imNum in range(1,numTiles):
                cumPhaseXY[imNum,:] = sum(TDvec[imagePath[imNum,range(numSteps[imNum])],:],0)    

        phaseXY = pixXYrel+cumPhaseXY

    # 
    # Finnally zero out the corrdinate system for assembling the images.
    minPhaseXY = np.amin(phaseXY,0)
    phaseXYcor = np.array((phaseXY[:,0]-minPhaseXY[0],phaseXY[:,1]-minPhaseXY[1])).T
  
    breakpoint()
    #  TODO:  Check displacements and make sure they average out across all directions to other images.
                        
    imPix = phaseXYcor.astype('int')
    # Determin size of stitched image
    stitchDim = np.amax(imPix,0)+imDim[0:2]
    
    # Create dictionary to store stitched image arrays
    anImage['Bf'] = np.zeros(stitchDim, dtype = 'uint16')
    stitchDevisor = np.zeros(stitchDim, dtype = 'uint16')
    imIter = 0
    anImage['imageLabels'] = {'Bf':0}
    
    # Generate a stitchDevisor...this is a matrix that gives the number of
    # times an individual pixel is overlapped so that each pixes is averaged
    # together appropriately.

    for m in range(numTiles):
        sy, ey = imPix[m,0], imPix[m,0]+imDim[0]
        sx, ex = imPix[m,1], imPix[m,1]+imDim[1]
        stitchDevisor[sy:ey,sx:ex] = stitchDevisor[sy:ey,sx:ex]+1                 


    imIter = 0
    for key in imageData.keys():
        if isinstance(imageData[key], dict) and 'imageStack' in imageData[key].keys():
            imIter+=1# timePoints = roiVals['timePoints']
    # objectArea = roiVals['objectArea']
    # figPlotGCs(timePoints, objectArea)
    # roiData = roiProcess(odDir, wellLbl, background)

            anImage['imageLabels'].update({key : imIter})
            anImage[key] = np.zeros(stitchDim, 'uint16')
            for m in range(numTiles):
                sy, ey = imPix[m,0], imPix[m,0]+imDim[0]
                sx, ex = imPix[m,1], imPix[m,1]+imDim[1]
                try:
                    imagedata = imageData[key]['imageStack'][:,:,m]/stitchDevisor[sy:ey,sx:ex]
                    anImage[key][sy:ey,sx:ex]= imagedata+anImage[key][sy:ey,sx:ex] 
                except:
                    breakpoint()
    
    ###################
    # OutPut Keys
    ###################

    anImage['stitchMeta'] = {}
    anImage['stitchMeta']['imPix']     = imPix
    anImage['stitchMeta']['xyzTime']   = xyzTime[5,:]
    anImage['stitchMeta']['minDist']   = minDist
    anImage['stitchMeta']['stitchDim'] = stitchDim

    return anImage

def assembleImage(imageFileName, pixSize, magnification, background, imPix = None, theta = None):
   
    imageData = parseImage(imageFileName)
    anImage = {}
    anImage['imageLabels'] = {}
    # Get stage position values for each image in the group
    xyzTime = imageData['Bf']['xyzTime']
    imDim   = imageData['Bf']['imageStack'].shape
    numTiles = imDim[2]
    rawImage = np.zeros((imDim), dtype = 'uint16')

    if theta is not None:
        rot_matrix = np.array([[np.cos(theta), np.sin (theta)],
                               [-np.sin(theta), np.cos(theta)]])
        
        # Correct Stage positions and set them to a minimum coordiante axis.  
        minXYZ = xyzTime[0,0:2]
        relXY  = np.array((xyzTime[:,1]-minXYZ[1],xyzTime[:,0]-minXYZ[0])).T
        pixXYrel = abs(relXY) * magnification / pixSize
        
        xy_n = rot_matrix @ pixXYrel.T
        phaseXY = xy_n.T
        minPhaseXY = np.amin(phaseXY,0)
        phaseXYcor = np.array((phaseXY[:,0]-minPhaseXY[0],phaseXY[:,1]-minPhaseXY[1])).T                
        imPix = phaseXYcor.astype('int')
   
    # Load Background Image if it exists
    if not hasattr(background, "shape"):
        background = np.full((imDim[0:2]), 0, dtype = 'uint16')
    
    #Read in images into RawStack and correct for background
    # for imNum in range(numTiles):
    #     # rawImage[:,:,imNum] = imageData['Bf']['imageStack'][:,:,imNum]-background
    #     imageData['Bf']['imageStack'][:,:,imNum]-=background
    
    
    # perform fft2 for all images in the stack in prep for alignment.
     
    anImage['fTrans'] = np.fft.fft2(imageData['Bf']['imageStack'][:,:,5]) # Save this transform 

    # Determin size of stitched image
    stitchDim = np.amax(imPix,0)+imDim[0:2]
    
    # Create dictionary to store stitched image arrays
    anImage['Bf'] = np.zeros(stitchDim, dtype = 'uint16')
    stitchDevisor = np.zeros(stitchDim, dtype = 'uint16')

    for m in range(numTiles):
        sy, ey = imPix[m,0], imPix[m,0]+imDim[0]
        sx, ex = imPix[m,1], imPix[m,1]+imDim[1]
        stitchDevisor[sy:ey,sx:ex] = stitchDevisor[sy:ey,sx:ex]+1                 
    
    imIter = 0
    for key in imageData.keys():
        if isinstance(imageData[key], dict) and 'imageStack' in imageData[key].keys():
            imIter+=1
            anImage['imageLabels'].update({key : imIter})
            anImage[key] = np.zeros(stitchDim, 'uint16')
            for m in range(numTiles):
                sy, ey = imPix[m,0], imPix[m,0]+imDim[0]
                sx, ex = imPix[m,1], imPix[m,1]+imDim[1]
                imagedata = imageData[key]['imageStack'][:,:,m]/stitchDevisor[sy:ey,sx:ex]
                anImage[key][sy:ey,sx:ex]= imagedata+anImage[key][sy:ey,sx:ex] 
   
    anImage['centIm'] = imageData['Bf']['imageStack'][:,:,5].squeeze()
    
    anImage['stitchMeta'] = {}
    anImage['stitchMeta']['imPix']   = imPix
    anImage['stitchMeta']['xyzTime'] = xyzTime[5,:]
    
    return anImage

def labelWell():

    return None

def thresholdImage(image, offset, coarseness):

    '''
    threshold intinsity image by subsampling the image background
    image:       uint16 mxn array
    offset:      float ratio that threshold value is increased over peak of subsampled max histogram
    coarseness:  int   size of subsample grid in n*x*n pixels

    '''
    
    xyDim = image.shape
    max_pix_hist = np.zeros(2**16, dtype = 'uint16')

    # Create row and column vectors to subsample image    
    rc_i = np.array([[r,c] for r in np.arange(start = 0, stop = xyDim[0], step = coarseness) for c in np.arange(start = 0, stop = xyDim[1], step = coarseness)], dtype = 'int')
    rc_e = rc_i + coarseness
    for (ri,ci), (re, ce) in zip(rc_i, rc_e):
        
        imseg = image[ri:re, ci:ce]
        max_pix = np.max(imseg)
        max_pix_hist[max_pix] += 1
      
    # Find max value of histogram segments
    k_segment = np.argmax(max_pix_hist[1:])+1
    
   
    if offset == None:
        thresholdValue = k_segment
    else:
        # now that value is a bit off so nudge it up a bit.
        maxind        = 1 #np.argmax(imageHist[1:])
        maxoffset     = abs(k_segment-maxind)+1
        thresholdValue = int(offset*maxoffset+maxind)
     
    return  thresholdValue

def SobelGradient(image):
    '''
    Apply a Sobel gradient filter to an image and return the 
    magnitude of the Sobel Graident

    input:  uint16 image
    output: uint16 image 
    '''

    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    gradientImage = (sobelx**2+sobely**2)**0.5

    return gradientImage.astype('uint16') 

def morphImage(inputImage, kernalerode, kernalopen, thresholdVal):
    '''
    Perform morphological opening and closing operations on binary images 
    inputImage:   uint16 array
    kernalerode:  int  the size of the kernal used to erode features
    kernalopen:   int  the size of a kernal used to dialate features
    thersholdVal: int  value used to binarize the inputImage
    '''

    bwBf = np.greater(inputImage, thresholdVal).astype('uint8')

    ekernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalerode, kernalerode))
    okernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernalopen , kernalopen))
    bwBf     = cv2.dilate(bwBf, ekernel, iterations = 1)
    bwBf     = cv2.erode( bwBf, ekernel, iterations = 1)
    bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_OPEN, okernel)
    bwBf     = cv2.morphologyEx(bwBf, cv2.MORPH_CLOSE,okernel)

    bwBf[1, :] = 1
    bwBf[:, 1] = 1
    bwBf[:,-1] = 1
    bwBf[-1,:] = 1

    return bwBf

def getRoiFileList(imagePath, roiID):
    ''' 
    Input
    --------
    odDir : python string indicating the ODELAY directory

    roiID:  A list of strings for the region of interest Identifiers these 
            should be folder names

    returns:  Dictionary of the subdirectorys and image file lists

    '''
    odelayPath = pathlib.Path(imagePath)
    
    odRoi = [roi for roi in odelayPath.iterdir() if roi.is_dir() and roi.name in roiID]

    roiDic = {}
    n = 0
    for roi in odRoi:
        roiDic[roi] = n
        n +=1

    expFileStructure = {}
    for subDir in odRoi:
        tempList = [imFile.name for imFile in odRoi[0].iterdir() if imFile.is_file() and ((imFile.suffix == '.hdf5') | (imFile.suffix == '.mat')) ]
        tempList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        
        tempDict = {}
        n = 0
        for roi in tempList:
            tempDict[roi] = n
            n += 1


        expFileStructure[subDir.name] = tempDict

    return expFileStructure

def diffsum(x):
    
    return np.sum(np.abs(np.diff(x)))

if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress = True, linewidth = 300)
    odDir = pathlib.Path('D:\\Python_Projects\\Projects and Data\\2021-09-21 Test ODELAY')
    expIndexPath = odDir / '2021_09_08_Mabs_24COstrains_Nodrug_2chambers_Index_ODELAYData.hdf5'
    imagePath = odDir

    roiID = 'H09'

    '''
        Data from Experiment Dictionary or Object
    '''
   
    expData = fio.loadData(expIndexPath)

    roiList = [*expData['roiFiles']]

    #####################################
    # Load Dictionary variables  There has to be a way to dynamically add these
    #####################################
    background       = expData['backgroundImage']
    defaultFitRanges = expData['defaultFitRanges']
    maxObj           = expData['maxObj']
    numTimePoints    = expData['numTimePoints']  # number of timeponts
    timerIncrement   = expData['timerIncrement'] # timer increment in seconds
    threshold_offset = expData['threshold_offset']
    pixSize          = expData['pixSize']
    sensorSize       = expData['sensorSize']
    magnification    = expData['magnification']
    coarseness       = expData['coarseness']
    kernalerode      = expData['kernalerode']
    kernalopen       = expData['kernalopen']
    roiFiles         = expData['roiFiles']
    experiment_name  = expData['experiment_name']
       
    '''
    The following code is to initialize data for all wells
    '''

    # if isinstance(roiID, str):
    #     roiLabel = roiID

    # elif isinstance(roiID, int):
    #     roiList = [*roiFiles]
    #     roiLabel = roiList[roiID]
    # # Else this will crash
    
    # roiPath = imagePath /  roiLabel
    # imageFileList = os.listdir(roiPath)
    # # Understand this gem of a regular expression sort.
    # imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
  
    for roiID in roiList:
        [imageIndex, imageFileList] = checkFocus(imagePath, expData, roiID)
        print(f'{roiID} has {len(imageFileList)} and {np.sum(imageIndex)}')
