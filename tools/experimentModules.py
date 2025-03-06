
import json
import numpy as np
import pandas as pd
import os
import pathlib
import sys

'''
This file will need updates as additional experiment types are created.  
'''


def moduleSelector(moduleName, expDict=None):

    if moduleName == 'ODELAY 96 Spot':
        
        expDict = microscopeDefaultConfig(expDict)
        
    elif moduleName == 'ODELAY 5 Condition':

        expDict = multiChamberConfig(expDict)

    elif moduleName == 'Macrophage 24 well':

        expDict = macrophageConfig(expDict)

    elif moduleName == 'External File':

        print('need to write a loader for this.  Using Defualt Config')
        expDict = loadModuleFile(expDict)

    return expDict

def twoConditionConfig(expDict=None, writeConfig = False):

    if not isinstance(expDict,dict):
        expDict = {}

    expDict['grid'] = {}
    expDict['grid']['rowName'] = ['E','F','G','H','I','J','K','L']
    expDict['grid']['colName'] = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19']
    expDict['grid']['wellSpacing']  =  4500
    expDict['grid']['wellDiameter'] =  4500 

    wellSpacing = expDict['grid']['wellSpacing']
    numRows = len(expDict['grid']['rowName'])
    numCols = len(expDict['grid']['colName'])

    expDict['grid']['sceneDim']=[wellSpacing*(numCols+1), wellSpacing*(numRows+1)]
    expDict['grid']['xgrid'] = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()
    expDict['grid']['ygrid'] = np.arange(0,wellSpacing*numRows,wellSpacing, dtype='float').tolist()

    expDict['numWells'] = len(expDict['grid']['rowName'])*len(expDict['grid']['colName']) # convert to method

    expDict['stageXYPos'] = np.zeros((expDict['numWells'], 2),dtype = 'float').tolist()

    # columnList = np.arange(1,numCols-1)
    columnList  = np.array([ 0, 1, 2, 3, 4, 5, 8, 9,10,11,12,13], dtype = 'int')
    roiOrder  = []    
    stageXYPos = {}
    roiDict    = {}
    cnt = 0
    order = 1
    for row, rowID in enumerate(expDict['grid']['rowName'],0):
        for col, colID in enumerate(expDict['grid']['colName'],0):
        
            stageXYPos[f'{rowID}{colID}']       = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row], cnt]
            roiDict[f'{rowID}{colID}'] = {}
            roiDict[f'{rowID}{colID}']['xyPos'] = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row]]
            roiDict[f'{rowID}{colID}']['ImageConds'] = {'Bf': 5}
            cnt+=1

        columnVec = columnList[::order]+row*numCols
        roiOrder.extend(columnVec.tolist())
        order *= -1

    roiList  =[[*roiDict][roiInd] for roiInd in roiOrder]
    print(roiList)

    expDict['roiOrder']   = roiOrder
    expDict['stageXYPos'] = stageXYPos
    expDict['roiDict']    = roiDict
    expDict['roiList']    = roiList
    
    expDict['iterNum']  = 0
    expDict['roiIndex'] = 0
    expDict['roi'] = expDict['roiIndex']
    
    expDict['numTimePoints'] = 500
    expDict['totIter']      = 500
    expDict['iterPeriod']   = 0
    expDict['getFileFlag']  = True
    expDict['ErrorOccured'] = False
    expDict['restartCnt']   = 0

    expDict['numTiles']   = 9
    
    expDict['mag'] = 10
    expDict['overlapPerc'] = 0.2
    expDict['pixSize']  = 6.45
    expDict['tileOrder']= [[-1, -1],
                           [ 0, -1],
                           [ 1, -1],
                           [-1,  0], 
                           [ 0,  0],
                           [ 1,  0],
                           [-1,  1],
                           [ 0,  1],
                           [ 1,  1]]

    #  Microscope Conditions              
    expDict['XYZOrigin'] = [35087,26180 , 6919];
    expDict['ExternModLoaded'] = False      
    expDict['FocusCond'] = 'Bf' 
    
    filePath = pathlib.Path(__file__).parent
    microscopeConfigPa=  filePath / 'microscopeInit.mmconfig'

    if writeConfig:
        with open(microscopeConfigPath, 'w') as fileOut:
            json.dump(expDict, fileOut)

    return expDict

def microscopeDefaultConfig(expDict=None, writeConfig = False):

    if not isinstance(expDict,dict):
        expDict = {}

    expDict['grid'] = {}
    expDict['grid']['rowName'] = ['E','F','G','H','I','J','K','L']
    expDict['grid']['colName'] = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19']
    expDict['grid']['wellSpacing']  =  4500
    expDict['grid']['wellDiameter'] =  4500 

    wellSpacing = expDict['grid']['wellSpacing']
    numRows = len(expDict['grid']['rowName'])
    numCols = len(expDict['grid']['colName'])

    expDict['grid']['sceneDim']=[wellSpacing*(numCols+1), wellSpacing*(numRows+1)]
    expDict['grid']['xgrid'] = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()
    expDict['grid']['ygrid'] = np.arange(0,wellSpacing*numRows,wellSpacing, dtype='float').tolist()

    expDict['numWells'] = len(expDict['grid']['rowName'])*len(expDict['grid']['colName']) # convert to method

    expDict['stageXYPos'] = np.zeros((expDict['numWells'], 2),dtype = 'float').tolist()

    columnList = np.arange(1,numCols-1)
    roiOrder  = []    
    stageXYPos = {}
    roiDict    = {}
    cnt = 0
    order = 1
    for row, rowID in enumerate(expDict['grid']['rowName'],0):
        for col, colID in enumerate(expDict['grid']['colName'],0):
        
            stageXYPos[f'{rowID}{colID}']       = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row], cnt]
            roiDict[f'{rowID}{colID}'] = {}
            roiDict[f'{rowID}{colID}']['xyPos'] = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row]]
            roiDict[f'{rowID}{colID}']['ImageConds'] = {'Bf': 5}
            cnt+=1

        columnVec = columnList[::order]+row*numCols
        roiOrder.extend(columnVec.tolist())
        order *= -1

    roiList  =[[*roiDict][roiInd] for roiInd in roiOrder]
    # print(roiList)

    expDict['roiOrder']   = roiOrder
    expDict['stageXYPos'] = stageXYPos
    expDict['roiDict']    = roiDict
    expDict['roiList']    = roiList
    
    expDict['iterNum']  = 0
    expDict['roiIndex'] = 0
    expDict['roi'] = expDict['roiIndex']
    
    expDict['numTimePoints'] = 500
    expDict['totIter']      = 500
    expDict['iterPeriod']   = 5*60*1000
    expDict['getFileFlag']  = True
    expDict['ErrorOccured'] = False
    expDict['restartCnt']   = 0

    expDict['numTiles']   = 9
    
    expDict['mag'] = 10
    expDict['overlapPerc'] = 0.2
    expDict['pixSize']  = 6.5
    expDict['tileOrder']= [[-1, -1],
                           [ 0, -1],
                           [ 1, -1],
                           [-1,  0], 
                           [ 0,  0],
                           [ 1,  0],
                           [-1,  1],
                           [ 0,  1],
                           [ 1,  1]]

    #  Microscope Conditions              
    expDict['XYZOrigin'] = [35087,26180 , 6919];      
    
    filePath = pathlib.Path(__file__).parent
    microscopeConfigPa=  filePath / 'microscopeInit.mmconfig'

    if writeConfig:
        with open(microscopeConfigPath, 'w') as fileOut:
            json.dump(expDict, fileOut)

    return expDict

def multiChamberConfig(expDict=None, writeConfig = False):

    if not isinstance(expDict,dict):
        expDict = {}

    expDict['grid'] = {}
    expDict['grid']['rowName'] = ['E','F','G','H','I','J','K','L']
    expDict['grid']['colName'] = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19']
    expDict['grid']['wellSpacing']  =  4500
    expDict['grid']['wellDiameter'] =  4500 

    wellSpacing = expDict['grid']['wellSpacing']
    numRows = len(expDict['grid']['rowName'])
    numCols = len(expDict['grid']['colName'])

    expDict['grid']['sceneDim']=[wellSpacing*(numCols+1), wellSpacing*(numRows+1)]
    expDict['grid']['xgrid'] = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()
    expDict['grid']['ygrid'] = np.arange(0,wellSpacing*numRows,wellSpacing, dtype='float').tolist()

    expDict['numWells'] = len(expDict['grid']['rowName'])*len(expDict['grid']['colName']) # convert to method

    expDict['stageXYPos'] = np.zeros((expDict['numWells'], 2),dtype = 'float').tolist()


    columnList = np.array([0,1,3,4,6,7,9,10,12,13])
    roiOrder  = []    
    stageXYPos = {}
    roiDict    = {}
    cnt = 0
    order = 1
    for row, rowID in enumerate(expDict['grid']['rowName'],0):
        for col, colID in enumerate(expDict['grid']['colName'],0):
        
            stageXYPos[f'{rowID}{colID}']       = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row], cnt]
            roiDict[f'{rowID}{colID}'] = {}
            roiDict[f'{rowID}{colID}']['xyPos'] = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row]]
            roiDict[f'{rowID}{colID}']['ImageConds'] = {'Bf': 5}
            cnt+=1

        columnVec = columnList[::order]+row*numCols
        roiOrder.extend(columnVec.tolist())
        order *= -1


    roiList  =[[*roiDict][roiInd] for roiInd in roiOrder[:5]]

    expDict['stageXYPos'] = stageXYPos
    expDict['roiDict']    = roiDict
    expDict['roiOrder']   = roiOrder
    expDict['roiList']    = roiList

    expDict['numTimePoints'] = 500
    expDict['iterNum']  = 0
    expDict['roiIndex'] = 0
    expDict['roi'] = expDict['roiIndex']

    expDict['totIter']      = 500
    expDict['iterPeriod']   = 1800*1000
    expDict['getFileFlag']  = True
    expDict['ErrorOccured'] = False
    expDict['restartCnt']   = 0

    expDict['numTiles']   = 9
    expDict['tileOrder']= [[-1, -1],
                        [ 0, -1],
                        [ 1, -1],
                        [-1,  0], 
                        [ 0,  0],
                        [ 1,  0],
                        [-1,  1],
                        [ 0,  1],
                        [ 1,  1]]

    #  Microscope Conditions              
    expDict['XYZOrigin'] = [35087,26180 , 6919];      
    
    filePath = pathlib.Path(__file__).parent
    microscopeConfigPath =  filePath / 'microscopeInit.mmconfig'

    if writeConfig:
        with open(microscopeConfigPath, 'w') as fileOut:
            json.duexpDict(expDict, fileOut)

    return expDict

def macrophageConfig(expDict=None, writeConfig = False):
    
    if expDict==None or not isinstance(expDict,dict):
        expDict = {}
    
    #127.76mm x 85.48

    expDict['grid'] = {}
    expDict['grid']['rowName']      = ['A','B','C','D']
    expDict['grid']['colName']      = ['01','02','03','04','05','06']
    expDict['grid']['wellSpacing']  = 19500
    expDict['grid']['xoffset']      = round(1950/2)
    expDict['grid']['yoffset']      = round(1950/2)
    expDict['grid']['wellDiameter'] = 19500 

    wellSpacing = expDict['grid']['wellSpacing']
    numRows = len(expDict['grid']['rowName'])
    numCols = len(expDict['grid']['colName'])
    
    expDict['grid']['sceneDim']=[wellSpacing*(numCols+1), wellSpacing*(numRows+1)]
    expDict['grid']['xgrid'] = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()
    expDict['grid']['ygrid'] = np.arange(0,wellSpacing*numRows,wellSpacing, dtype='float').tolist()

    expDict['numWells'] = len(expDict['grid']['rowName'])*len(expDict['grid']['colName']) # convert to method

    expDict['stageXYPos'] = np.zeros((expDict['numWells'], 2),dtype = 'float').tolist()

    columnList = np.arange(0,numCols)
    roiOrder   = []    
    stageXYPos = {}
    roiDict    = {}
    cnt = 0
    order = 1
    for row, rowID in enumerate(expDict['grid']['rowName'],0):
        for col, colID in enumerate(expDict['grid']['colName'],0):
        
            stageXYPos[f'{rowID}{colID}']       = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row], cnt]
            roiDict[f'{rowID}{colID}'] = {}
            roiDict[f'{rowID}{colID}']['xyPos'] = [expDict['grid']['xgrid'][col], expDict['grid']['ygrid'][row]]
            roiDict[f'{rowID}{colID}']['ImageConds'] = {'Bf': 5}
            cnt+=1

        columnVec = columnList[::order]+row*numCols
        roiOrder.extend(columnVec.tolist())
        order *= -1
    
    roiList  =[[*roiDict][roiInd] for roiInd in roiOrder]

    expDict['stageXYPos'] = stageXYPos
    expDict['roiDict']    = roiDict
    expDict['roiOrder']   = roiOrder
    expDict['roiList']    = roiList

    expDict['iterNum']  = 0
    expDict['roiIndex'] = 0
    expDict['roi'] = expDict['roiIndex']

    expDict['totIter']       = 96
    expDict['numTimePoints'] = 320
    expDict['iterPeriod']    = 1800*1000
    expDict['getFileFlag']   = True
    expDict['ErrorOccured']  = False
    expDict['restartCnt']    = 0

    expDict['numTiles']   = 9
    
    expDict['tileOrder']= [[-1, -1],
                        [ 0, -1],
                        [ 1, -1],
                        [-1,  0], 
                        [ 0,  0],
                        [ 1,  0],
                        [-1,  1],
                        [ 0,  1],
                        [ 1,  1]]

    #  Microscope Conditions              
    expDict['XYZOrigin'] = [35087,26180 , 6919];      
    
    # Load dictionary that will define microscope states based on configuration loaded.
    # Brightfield Image Conditions

    filePath = pathlib.Path(__file__).parent
    microscopeConfigPath =  filePath / 'microscopeInit.mmconfig'

    if writeConfig:
        with open(microscopeConfigPath, 'w') as fileOut:
            json.duexpDict(expDict, fileOut)

    return expDict

def loadModuleFile(expDict = None, writeConfig = False):

    if expDict==None or not isinstance(expDict,dict):
        expDict = {}

    fileName = pathlib.Path('N:\ODELAM Project\ODELAM 2019-01-17 TB MacroPhage Development\ODELAY Test Spot-Layout.xlsx')

    if fileName == None:
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Spot-Layout File", "","(*Spot-Layout.xlsx);;", options=options)

    spotlayoutPath = pathlib.Path(fileName)
    
    if spotlayoutPath.exists():
        columnID = 'Strain ID'
        strainID = pd.read_excel(spotlayoutPath, sheet_name='Sheet1', header=0, usecols="A:L").set_index('ROI')

        experimentROI = pd.read_excel(spotlayoutPath, sheet_name='Sheet2', header=0, usecols="A:D").set_index('ROI')
        roiList = experimentROI.index.tolist()

        roiDict = {}        
        stageXYPos = {}
        roiOrder = []

        experimentROI = pd.read_excel(spotlayoutPath, sheet_name='Sheet2', header=0, usecols="A:D").set_index('ROI')
        roiList = experimentROI.index.tolist()
        for cnt, roi in enumerate(roiList, 0):
           
            stageXYPos[roi]       = [experimentROI.loc[roi,'xPos'], experimentROI.loc[roi,'yPos'], cnt]
            roiDict[roi] = {}
            roiDict[roi]['xyPos'] = [experimentROI.loc[roi,'xPos'], experimentROI.loc[roi,'yPos']]
            roiDict[roi]['ImageConds'] = experimentROI.loc[roi,'Image Modes']
            roiOrder.append(cnt)
       

        expDict['stageXYPos'] = stageXYPos
        expDict['roiDict']    = roiDict
        expDict['roiOrder']   = roiOrder
        expDict['roiList']    = roiList

        return expDict
      
if __name__ == "__main__":
    moduleName = 'External File'
    mP = moduleSelector(moduleName, expDict=None)

    print(mP)


