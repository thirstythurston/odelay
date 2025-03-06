# Functions to load ODELAY generated MATFILES

# Internal Libraries
import click
import csv
from datetime import datetime
import json
import os
import pathlib
import re
import sys
import time

# External Libraries
import cv2
from fast_histogram import histogram1d
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tools.imagepl    as opl
import tools.odelayplot as odp
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QFileDialog
from PyQt5.QtCore    import QDir

def getStrings(str_array):
    '''Make a str_array into character list'''

    return ''.join(chr(c) for c in str_array)

def getHDF5str(str_array):
    '''Generate a string array from individual bchar values'''

    return ''.join(c.astype(str)  for c in str_array.squeeze())

def _mat_check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _mattodict(d[key])
    return d

def _mattodict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[strg] = _mattodict(elem)
 
        else:
            d[strg] = elem
    return d

def _mattolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, sio.matlab.mio5_params.mat_struct):
            elem_list.append(_mattodict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_mattolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

def _mat_parseData(group_Obj,  hd5_file):
    '''
    Matlab -v7.3 data parser.  If the data is a object then it could be a cell array or a nested structure.
    '''
    group_Attrs = dir(group_Obj)
    attrs_Keys = [key for key in group_Obj.attrs.keys()]
    numeric_list = ['double', 'uint8','uint16','logical','int8', 'int16']

    if 'dtype' in group_Attrs:
        # First test for Objects if the object reference 
        if group_Obj.dtype == 'object':
            data = _matcell_todict(group_Obj,hd5_file)

        elif 'MATLAB_class' in attrs_Keys:
            # Luckily they put the MATLAB_class in attributes so that can define the data type as numeric char or cells
            group_attrs = group_Obj.attrs['MATLAB_class'].decode('utf-8')
            data = []

            if group_attrs in numeric_list:
                data.append(np.array(group_Obj, dtype = group_Obj.dtype).squeeze())

            elif group_attrs == 'char':
                data.append(getStrings(group_Obj))
            
            elif group_attrs == 'cell':
                del data
                data = {}
                data = _matcell_todict(group_Obj, hd5_file) 
  
    return data

def _mat_hd5ToDict(group_Obj, hd5_file):
    '''
    Import MATLAB hd5f files from MATLAB -v7.3.  This will parse the data into nested python dictionaries 
    '''
    group_Dict = {}
    for name, elem in group_Obj.items():
    # Iterate through items and check if they are groups or datasets
 
        if type(elem) is h5py.Group:
            group_Dict[name] = _mat_hd5ToDict(elem, hd5_file)
      
        elif type(elem) is h5py.Dataset:
            group_Dict[name] = _mat_parseData(elem, hd5_file)

    return group_Dict

def _matcell_todict(group_Obj,hd5_file):

    objShape = group_Obj.shape

    if (objShape[0] == 1 or objShape[1] == 1):
        data= []
        for objRef in group_Obj:
            for ref_elem in objRef:
                if hd5_file[ref_elem].dtype == 'object':
                    data.append(_matcell_todict(hd5_file[ref_elem], hd5_file))
                else:
                    data.append(_mat_parseData(hd5_file[ref_elem],  hd5_file))

    else:
        data = {}
        for row in range(objShape[1]):
            name = getStrings(hd5_file[group_Obj[0][row]])
            str_array = []
            
            for col in range(1,objShape[0]):
                str_array.append(getStrings(hd5_file[group_Obj[col][row]]))
            
            data[name] = str_array

    return data

def _decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def _saveDict(dic, hdf5Obj):
    """
    Recursively save a python dictionary to a HDF5 file.  
    input:  dic - is the dictionary to save
            hdf5Obj - is the current hdf5 group or file handle to save the data to.    
    """
    for key, item in dic.items():

        
        if isinstance(item, (np.ndarray, np.int32, np.uint16, np.int64, np.float64)):

            hdf5Obj.create_dataset(key, data = item)

        elif isinstance(item, (bytes, bool, int, float)):

            hdf5Obj.create_dataset(key, data = np.array(item, dtype=type(item)))


        elif isinstance(item, str):
            # breakpoint()
            asciiList = [el.encode("ascii", "ignore") for el in item]
            hdf5Obj.create_dataset(key, (len(asciiList),1),'S10', asciiList)

        elif isinstance(item, dict):
            # if the item is a dictionary, create a new hdf5 group and then recursively continue
            grpObj = hdf5Obj.create_group(key)
            _saveDict(item, grpObj)

        else:
            raise ValueError('Cannot save %s of %s type' %(key,type(item)))

def _getHDF5Data(hdf5Obj):
    """
    Recursively load data from HDF5 file.  
    """
    d = {}
    for key, item in hdf5Obj.items():
 
        if   isinstance(item, h5py.Dataset):
            if item.dtype == 'S10':
                d[key] = getHDF5str(item[()])
            else: 
              
                d[key] = item[()]
          
        elif isinstance(item, h5py.Group):
            d[key] = _getHDF5Data(item)

    return d

def loadmatlab(filename):
    '''
    loadmat opens  MATLAB® formatted binary file (MAT-file) 
    loadmat should be called instead of direct sio.loadmat because it recoveres 
    python dictionaries from mat files. It calls the function _check_keys to curate all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True, verify_compressed_data_integrity=False)
    return _mat_check_keys(data)

def math5py(filename):
    '''
    Load HDF5 saved file in ODELAY.  Not all files are this type so it tends 
    to be problematic
    '''
    f = h5py.File(filename,'r') 
    d = _mat_hd5ToDict(f,f)
    f.close()
    return d

def saveDict(fileName, saveDic):

    with h5py.File(fileName,'w') as hdf5Obj:
        _saveDict(saveDic, hdf5Obj)

def saveROI(fileLocation, roiDict):
    '''Steps to save a file:
        1.  Save HFD5 file from entered dictionary
    '''
    filePath = pathlib.Path(fileLocation) 
    if 'roiLabel' in roiDict.keys():

        fileName = filePath /(roiDict['roiLabel'] +'.hdf5')

        with h5py.File(fileName,'w') as hdf5Obj:
            _saveDict(roiDict, hdf5Obj)
            
    return None

def loadData(filename):
    """
    Load data from HDF5 file into a python Dictionary
    This function will attempt to load MATLAB *.mat files based on thier 
    file names.
    There maybe problems with the dictionary returned in that it may need 
    to be squeezed.
    """
    d = {}

    if '.mat' in filename.name:
        
        try:
            d = math5py(filename)  
        except:
            d = loadmatlab(filename)

    elif '.hdf5' in filename.name:
        with h5py.File(filename, 'r') as hdf5Obj:
            d =  _getHDF5Data(hdf5Obj)
            
    return d

def summarizeMatLabExp(dataPath, saveSwitch):
    expDict = {}
    if isinstance(dataPath, str):
        dataPath = pathlib.Path(dataPath)
    else:
        dataPath = dataPath
    
    checkPath = dataPath / 'ODELAY Well Data'
    
    expIndexPath = list(dataPath.glob('*Index_ODELAYData.mat'))
    
    if pathlib.Path.exists(checkPath):
        expData = loadData(expIndexPath[0])
 
        expName = expData['ExperimentName'][0]
        savePath = dataPath /f'{expName} summary.hdf5'

        fileList = sorted(checkPath.glob('*.mat'))
    
        expDict = {}
        with click.progressbar(fileList) as fileBar:
            for dataFile in fileBar:

                roiData = loadData(dataFile)

                roi =  roiData['WellDataTemp']['WellID']
                if 'FitDataGompDT' in list(roiData['Tracks2Temp']['ObjectInfo'].keys()):
                    fitDataKey  = 'FitDataGompDT'
                    colList = ['a', 'b', 't-lag', 'dT', 'fssq', 'Tlag', 'Td', 'Tex', 'ATex', 'Aplateau', 'TdFlag', 'TexFlag', 'TVmax', 'Tplat','exitflag','fssq per obs','empty']
                    fitDataCols = {colList[val] : val for val in range(len(colList))}
            
                elif 'FitDataGomp' in list(roiData['Tracks2Temp']['ObjectInfo'].keys()):
                    fitDataKey  = 'FitDataGomp'
                    colList = ['a', 'b', 'vmax', 't-lag', 'fssq', 'Tlag', 'Td', 'Tex', 'ATex', 'Aplateau', 'TdFlag', 'TexFlag', 'TVmax', 'Tplat','exitflag','fssq per obs','empty']
                    fitDataCols = {colList[val] : val for val in range(len(colList))}

                elif 'FitData' in list(roiData['Tracks2Temp']['ObjectInfo'].keys()):
                    fitDataKey  = 'FitData'
                    colList = ['a', 'b', 'c', 'd', 'fssq', 'Tlag', 'Td', 'Tex', 'ATex', 'Aplateau', 'TdFlag', 'TexFlag', 'TVmax', 'Tplat','exitflag','fssq per obs','empty']
                    fitDataCols = {colList[val] : val for val in range(len(colList))}

                rc = roiData['Tracks2Temp']['ObjectInfo'][fitDataKey].shape
        
                idVec = np.arange(rc[0], dtype = 'uint32')
                inds = ~np.isnan(roiData['Tracks2Temp']['ObjectInfo'][fitDataKey][:,1])
                
                roiDict = {}
                try:
                    roiDict['fitDataCols']= fitDataCols
                    roiDict['objectArea'] = roiData['Tracks2Temp']['ObjectInfo']['ObjectArea'][inds,:]
                    roiDict['timePoints'] = roiData['Tracks2Temp']['ObjectInfo']['TimePoints']
                    roiDict['fitData']    = roiData['Tracks2Temp']['ObjectInfo'][fitDataKey][inds,:]
                    roiDict['objID']      = idVec[inds]
                except IndexError:
                    roiDict['fitDataCols']= fitDataCols
                    roiDict['objectArea'] = roiData['Tracks2Temp']['ObjectInfo']['ObjectArea']
                    roiDict['timePoints'] = roiData['Tracks2Temp']['ObjectInfo']['TimePoints']
                    roiDict['fitData']    = roiData['Tracks2Temp']['ObjectInfo'][fitDataKey]
                    roiDict['objID']      = idVec
                

                roiDict['roi']        = roi
                roiDict['roiInfo']    = {}
                    
                expDict[roi] = roiDict

        # spotlayoutPath = [*dataPath.glob('*ODELAYExpDisc.xlsx')]
        # if len(spotlayoutPath)==1:
           
        #     strainID = pd.read_excel(spotlayoutPath[0], sheet_name='Sheet1', header=29, usecols="B:J").set_index('ODELAY Well')
        #     columnID     = 'Strain ID'
        #     strainInfo1  = 'Plot Name'
        #     strainInfo2  = 'Misc1'
          
        #     for roi in expDict.keys():
            
        #         expDict[roi]['roiInfo'][columnID] = f'{strainID.loc[roi][columnID]}-{strainID.loc[roi][strainInfo1]}-{strainID.loc[roi][strainInfo2]}-{roi}'

        
        if saveSwitch:
            saveDict(savePath, expDict)

    else:
        print('Could Not Find ODELAY Data Folder')

    return expDict

def summarizeExp(dataPath, saveSwitch):

    if isinstance(dataPath, str):
        dataPath = pathlib.Path(dataPath)
    else:
        dataPath = dataPath

    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')

    expData = loadData(expIndexPath)

    expName = expData['experiment_name']
    savePath = dataPath /f'{expName} summary.hdf5'

    roiList = list(expData['roiFiles'].keys())
    roiList.sort()
   
    # generate nested dictionary of FitData, ObjectArea, and TimePoints
    expDict = {}
    with click.progressbar(roiList) as roiBar:
        for roi in roiBar:
            roiPath = dataPath / 'ODELAY Roi Data' / f'{roi}.hdf5'
            if roiPath.exists():
                roiData = loadData(roiPath)
                rc = roiData['fitData'].shape
                idVec = np.arange(rc[0], dtype = 'uint32')
                inds = roiData['fitData'][:,0]>0
                roiDict = {}
                roiDict['fitDataCols']= roiData['fitDataCols']
                roiDict['fitData']    = roiData['fitData'][inds,:]
                roiDict['objectArea'] = roiData['objectArea'][inds,:]
                roiDict['timePoints'] = roiData['timePoints']
                roiDict['numObj']     = roiData['numObj']
                roiDict['objID']      = idVec[inds]
                roiDict['roi']        = roi
                roiDict['roiInfo']    = {}

                expDict[roi] = roiDict


    spotlayoutPath = [*dataPath.glob('*Spot-Layout.xlsx')][0]
   
    # use the columns filled in the Spot-Layout to label the plots
    if hasattr(spotlayoutPath, 'name'):
        columnID = 'Strain ID'
        strainID = pd.read_excel(spotlayoutPath, sheet_name='Sheet1', header=0, usecols="A:L").set_index('ROI')
        columnList = strainID.columns.to_list()
        columnLabels = ['Strain ID','Strain Info 1','Strain Info 2', 'Media Condition 1', 'Media Condition 2', 'Misc 1', 'Misc 2']
        for roi in expDict.keys():
            strainInfo = ' '.join([str(strainID.loc[roi][colID]) for colID in columnLabels if str(strainID.loc[roi][colID])!='nan']) + ' ' + roi
            expDict[roi]['roiInfo'][columnID] = strainInfo
            
    if saveSwitch:
        saveDict(savePath, expDict)

    return expDict

def exportcsv(dataPath, roiList=None):
    ''' Export csv files of object area and fit data.  All data is exported including non-fit data.'''
    # TODO:  Add ability to filter data and export data label vectors.
    # TODO:  Add ability to look for summary data to shorten export time

    dateString = datetime.today().strftime("%Y-%m-%d")
    directoryName = f'{dateString}_csvOut'

    saveLocation = dataPath / directoryName
    if not saveLocation.exists():
        saveLocation.mkdir()
   
    summaryList = list(dataPath.glob('*summary.hdf5'))
    
    if len(summaryList)==1:
        expDict = loadData(summaryList[0])
        print('summary loaded')

        for roi in expDict.keys():
            roiData = expDict[roi]
           
            rc = roiData['fitData'].shape
            idVec = np.arange(rc[0], dtype = 'uint32')
            inds = roiData['fitData'][:,0]>0

            timePoints    = roiData['timePoints']
            objectArea    = roiData['objectArea'][inds,:]
     
            fitDataCols   = roiData['fitDataCols']
            fitDataHeader = [key for key, value in sorted(fitDataCols.items(), key=lambda item: item[1])]

            fitData       = roiData['fitData'][inds,:len(fitDataHeader)]

            fitDataFrame = pd.DataFrame(fitData,    columns = fitDataHeader)
            objAreaFrame = pd.DataFrame(objectArea, columns = timePoints)
            fitDataFrame['object ID'] = roiData['objID']
            
       
            objArea_csv_Path = saveLocation / f'{roi}-objectArea.csv'
            fitData_csv_Path = saveLocation / f'{roi}-FitData.csv'

            fitDataFrame.to_csv(fitData_csv_Path, index = None, header=True)
            objAreaFrame.to_csv(objArea_csv_Path, index = None, header=True)
         
    else:
        for roi in roiList:
            roiPath = dataPath / 'ODELAY Roi Data' / f'{roi}.hdf5'
            if roiPath.exists():
                roiData = loadData(roiPath)

                rc = roiData['fitData'].shape
                idVec = np.arange(rc[0], dtype = 'uint32')
                inds = roiData['fitData'][:,0]>0

                timePoints    = roiData['timePoints']
                objectArea    = roiData['objectArea'][inds,:]
                fitData       = roiData['fitData'][inds,0:15]
                fitDataCols   = roiData['fitDataCols']
                fitDataHeader = [key for key, value in sorted(fitDataCols.items(), key=lambda item: item[1])]
                
                fitDataFrame = pd.DataFrame(fitData,    columns = fitDataHeader)
                objAreaFrame = pd.DataFrame(objectArea, columns = timePoints)
                fitDataFrame['object ID'] = idVec[inds]
                objAreaFrame['object ID'] = idVec[inds]

                objArea_csv_Path = saveLocation / f'{roi}-objectArea.csv'
                fitData_csv_Path = saveLocation / f'{roi}-FitData.csv'

                fitDataFrame.to_csv(fitData_csv_Path, index = None, header=True)
                objAreaFrame.to_csv(objArea_csv_Path, index = None, header=True)
            else:
                print(f"{roi} did not process as its data file doesn't exist")

    return None

def exportavi(  imagepath, datapath, roi, objID = None):

    '''Write XVID encoded *.avi movie for timecourse images.'''


    dataPath = pathlib.Path(datapath)
    imagePath = pathlib.Path(imagepath)

    directoryName = 'ODELAY Roi AVI'

    saveLocation = dataPath / directoryName
    if not saveLocation.exists():
        pathlib.Path.mkdir(saveLocation)
    #  '''Write an AVI file that shows the ROI over time'''
    # TODO:  figure out way to zoom in on colony area, add time code, and scale bar

    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')

    expIndex = loadData(expIndexPath)

    roiPath = dataPath / 'ODELAY Roi Data' / f'{roi}.hdf5'
    roiData = loadData(roiPath)

    imageList = list(expIndex['roiFiles'][roi].keys())
    imageList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    numImage = len(imageList)
    # Determin size of stitched image

    stitchDim     = np.zeros((numImage,2), dtype = 'float')
    magnification = expIndex['magnification']
    pixSize       = expIndex['pixSize']
    background    = expIndex['backgroundImage']

    breakpoint()
    for ai in roiData['stitchMeta'].keys():
        stitchDim[int(ai),:] = roiData['stitchMeta'][ai]['stitchDim']

    vidDim = np.median(stitchDim,0).astype('uint')
   
    fps = 10.0

    vidFileName = str(saveLocation / f'{roi}.avi')

    fcc = cv2.VideoWriter_fourcc(*'XVID')

    aviOut = cv2.VideoWriter(vidFileName,fcc, fps, (vidDim[1],vidDim[0]),1)
    imInd = 0
    
    
    for im in imageList:

        # load image
        # rectify umageDims
        # Adjust Contrast to uint8
        # write frame in avi
        roiKey = f'{imInd:03d}'
        imPix = roiData['stitchMeta'][roiKey]['imPix']
        imageFilePath = imagePath / roi / im
        rawImage  = opl.assembleImage(imageFilePath, pixSize, magnification, background, imPix)

        # Generate histogram of the loaded image
        imageHist = histogram1d(rawImage['Bf'].ravel(),2**16,[0,2**16],weights = None).astype('float')
        # Calculate the cumulative probability ignoring zero values 
        cumHist = np.cumsum(imageHist)
        cumProb = (cumHist-cumHist[0])/(cumHist[2**16-1]-cumHist[0])
        # set low and high values ot normalize image contrast.        
        loval = np.argmax(cumProb>0.00001)
        hival = np.argmax(cumProb>=0.9995)

        rc = np.min((stitchDim[imInd,:],vidDim),axis=0).astype('int')
        adjIm = np.zeros((vidDim[0],vidDim[1]), dtype = 'float')
        adjIm[:rc[0],:rc[1]] = (rawImage['Bf'][:rc[0],:rc[1]].astype('float') - loval.astype('float'))/(hival.astype('float') - loval.astype('float'))*254
        lim = np.iinfo('uint8')
        scIm = np.clip(adjIm, lim.min, lim.max) 

        vidIm = np.stack([scIm, scIm, scIm], axis=2).astype('uint8')

        aviOut.write(vidIm)
        imInd +=1
      
    
    aviOut.release()
  

    
    return None

def exporttiffs(imagepath, datapath, roi, objID = None):
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

    indexList = [*dataPath.glob('*Index_ODELAYData.*')]

    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
    else:
        print('Could not find the correct index file or there were more than one in the diretory')

    expData = loadData(expIndexPath)
    #####################################
    # Load Dictionary variables  There has to be a way to dynamically add these
    #####################################
    background       = expData['backgroundImage']
    numTimePoints    = expData['numTimePoints']   # number of timeponts
    pixSize          = expData['pixSize']
    magnification    = expData['magnification']
    roiFiles         = expData['roiFiles']
    odelayDataPath   = dataPath / 'ODELAY Roi Data'

    # Else this will crash
    
    roiList = [*roiFiles]

    tiffPath = dataPath / 'ODELAY Tiff Images'
    if not tiffPath.exists():
        tiffPath.mkdir()
    
    if roi in roiList:
        roiPath = imagePath /  roi
        fileList = os.listdir(roiPath)
        imageFileList = [fileName for fileName in fileList if '.mat' in fileName]
        # Understand this gem of a regular expression sort.
        imageFileList.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        numImages = len(imageFileList)
        
        tiffRoiPath = tiffPath / roi
        if not tiffRoiPath.exists():
            tiffRoiPath.mkdir()

        # Start Processing Data Here
        for aI in range(numImages):

            imageFilePath = roiPath / imageFileList[aI]
        
            anImage = opl.stitchImage(imageFilePath, pixSize, magnification, background)
            
            for imlbl in anImage['imageLabels'].keys():
                saveFilePath =  tiffRoiPath / f'{roi}_{imlbl}_{aI+1:03d}.tiff'
                uint8Image = odp.scaleImage(anImage[imlbl])
                retVal = cv2.imwrite(str(saveFilePath), uint8Image)
        
            
    return None

def readExpDisc(dataPath):

    '''Reads formatted excell spreadsheet and returns a dataframe with the experiment orgainsation.'''

    spotlayoutPath = [*dataPath.glob('*Spot-Layout.xlsx')]
    if len(spotlayoutPath)==1:
        strainID = pd.read_excel(spotlayoutPath, sheet_name='Sheet1', header=0, usecols="A:L").set_index('ROI')

    return strainID

def setdatadir(loc_data_dir):
    '''Set the directory where processed ODELAY data is loaded/saved'''
    configfilePath = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

    with open(configfilePath, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localDataPath = pathlib.Path(loc_data_dir)

    if localDataPath.exists:
        resolvedPath = localDataPath.resolve()
        LocalDataPathstr  = str(resolvedPath)

        HPCDataPath  = LocalDataPathstr.replace('\\','/').replace('//helens','/')

        odelayConfig['LocalDataDir'] = loc_data_dir
        odelayConfig['HPCDataDir']   = HPCDataPath    
        print(f'Data Directory path from local computer is: {loc_data_dir}')
        print(f'Data Directory path from  HPC  computer is: {HPCDataPath}')
    

    odelayConfig['PathCheck'] = False

    with open(configfilePath, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return resolvedPath

def setimagedir(loc_image_dir):
    '''Set the directory where the experiment's images are located'''

    configfilePath = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

    with open(configfilePath, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localImagePath = pathlib.Path(loc_image_dir)
    
    if localImagePath.exists():
        
        resolvedPath  = pathlib.Path(loc_image_dir).resolve()
        LocalImagePathstr  = str(resolvedPath)
        
        HPCImagePath = LocalImagePathstr.replace('\\','/').replace('//pplhpc1ces','/gpfs/scratch')

        odelayConfig['LocalImageDir'] = loc_image_dir
        odelayConfig['HPCImageDir']   = HPCImagePath    
        print(f'Image Directory path from local computer is: {loc_image_dir}')
        print(f'Image Directory path from  HPC  computer is: {HPCImagePath}')


    odelayConfig['PathCheck'] = False
    
    with open(configfilePath, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)


    return resolvedPath

def loadConfig():
    configfilePath = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

    with open(configfilePath, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    return odelayConfig

def saveConfig(odelayConfig):

    configfilePath = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )
    odelayConfig['PathCheck'] = False

    with open(configfilePath, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return odelayConfig

def readMMConfigFile(filePath):

    configFilePath = pathlib.Path(filePath)

    configList = []

    configDict = {
            'Device':{},
            'Parent':{},
            'Label':{},
            'Group':{}
            }

    with open(configFilePath) as csvfile:
        reader = csv.reader(_decomment(csvfile))
        for row in reader:
            configList.append(row)

    for row in configList:
        
        if row[0] == 'Device':
            configDict['Device'].update({row[1]: [row[2],row[3]]})

        elif row[0] == 'Parent':
            configDict['Parent'].update({row[1]:row[2]})

        elif row[0] == 'Label':
            if row[1] in configList[row[0]].keys():
                configDict[row[0]][row[1]].update({row[2]:row[3]})

            else:
                configDict[row[0]].update({row[1]:{row[2]:row[3]}})

        elif row[0] == 'Group':
            configDict['Group'].update({row[1]:{row[2]: row[3]}})

    return configDict

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def openFileDialog():

    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    fileName, _ = QFileDialog.getOpenFileName(None,"Select ODELAY Data Set", "","ODELAYExpDisc (*Index_ODELAYData.mat);; Mat-Files (*.mat)", options=options)
    app.quit()
    return fileName