

import os
import pathlib
import re
import time
import sys
import json
import cv2
import h5py
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl

from scipy.sparse   import csr_matrix
from fast_histogram import histogram1d
from datetime       import datetime
from importlib      import reload

from PyQt5              import QtCore, QtGui, QtWidgets
# from PyQt5.QtMultimedia import QMediaPlayer
# from PyQt5.QtMultimedia import QMediaContent
# from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets    import QApplication, QMainWindow, QLabel, QSizePolicy, QWidget, QInputDialog, QFileDialog
from PyQt5.QtWidgets    import QHBoxLayout, QLabel, QPushButton, QStyle, QVBoxLayout, QWidget, QSlider, QPushButton, QAction
from PyQt5.QtGui        import QImage, QPixmap, QIcon
from PyQt5.QtCore       import QDir, Qt, QUrl

import tools.imagepl as opl
import tools.fileio as fio

def figPlotGCs(roiDict, organism='Yeast', saveAll=False, savePath=None):
    ''' Plots growth curves using matplotlib'''

    plt.close('all')

    pltRange = setPlotRange(organism)
    if 'roiLabel' in roiDict.keys():
        roiDict['roi'] = roiDict['roiLabel']

    roiID         = roiDict['roi']
    timePoints    = roiDict['timePoints']/pltRange['GCs']['devisor']
    rawobjectArea = roiDict['objectArea']
    rawfitData    = roiDict['fitData']

    numObsv = pltRange['GCs']['numObservations']
    rngTime = pltRange['GCs']['xRange']
    rngArea = pltRange['GCs']['yRange']
    rngTdbl = pltRange['Dbl']['xRange']
    rngTlag = pltRange['Lag']['xRange']
    rngTexp = pltRange['Tex']['xRange']
    rngNDub = pltRange['NumDbl']['xRange']
    
    if ('roiInfo' in roiDict.keys()) and (len(roiDict['roiInfo'])>0) :
        roiID = roiDict['roiInfo']['Strain ID']
     


    numObservations = np.sum(rawobjectArea>0, 1) > numObsv
    numDbl   = rawfitData[:,1]>0
    fitIndex = rawfitData[:,0]>0
    dataFilter = numObservations * fitIndex * numDbl

    if np.sum(dataFilter)<10:
        print('Data filter too strict. Reducing required timepoints to zero')
        numObservations = np.sum(rawobjectArea>0, 1) > np.round(numObsv/2)
        dataFilter = fitIndex * numDbl

        

    fitData    = rawfitData[dataFilter, :]
    objectArea = rawobjectArea[dataFilter,:].transpose()

    fitData[:,3]/=pltRange['Tex']['devisor']
    fitData[:,5]/=pltRange['Lag']['devisor']
    fitData[:,6]/=pltRange['Dbl']['devisor']

    textLbls= ['Growth Curves','Td (hrs)', 'Tlag (hrs)','Texp (hrs)','Num Dbl']
    lineColor = np.array([  [0,  0,   0, 0.3],
                            [0,  0,   1, 1],
                            [0,  0.7, 0, 1],
                            [1,  0,   0, 1],
                            [0.7,0.5, 0, 1]], dtype = 'float')

    xLim = np.array([rngTime,
                    rngTdbl,
                    rngTlag,
                    rngTexp,
                    rngNDub], dtype = 'float64')
    wScale = 0.75

    numbins   = 75
    fitCol = [6,6,5,3,1]
    normVirts = np.zeros((5,numbins), dtype='float64')
    virts     = np.zeros((5,numbins), dtype='float64')
    nbins     = np.zeros((5,numbins), dtype='float64')
    with np.errstate(divide='ignore',invalid='ignore'):
        for cnt in range(5):
            nbins[cnt,:] = np.linspace(xLim[cnt,0], xLim[cnt,1], num=numbins)
            virts[cnt,:] = histogram1d( fitData[:,fitCol[cnt]], 75, xLim[cnt,:], weights = None)
            normVirts[cnt,:] = (virts[cnt,:]/np.max(virts[cnt,2:-10]))*wScale 
        
        lgobjectArea = np.log2(objectArea)
        

    axesPos = np.array([[0.1875, 0.66666667, 0.75, 0.28],
                        [0.1875, 0.48666667, 0.75, 0.1],
                        [0.1875, 0.33333333, 0.75, 0.1],
                        [0.1875, 0.19333333, 0.75, 0.1],
                        [0.1875, 0.05333333, 0.75, 0.1]], dtype = 'float64')

    xLim = np.array([rngTime,
                    rngTdbl,
                    rngTlag,
                    rngTexp,
                    rngNDub], dtype = 'float64')

    yLim = np.array(   [rngArea,
                        [0,1],
                        [0,1],
                        [0,1],
                        [0,1]], dtype = 'float64')

    Label_Font = 12
    Title_Font = 12

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = Title_Font
    mpl.rcParams['axes.labelsize'] = Label_Font

    gcFig = plt.figure(figsize=[4,7.5], dpi=100, facecolor='w')
    axs = []
    n = 0
    axs.append(plt.axes(axesPos[n,:], xlim=xLim[n,:], ylim=yLim[n,:]))
    axs[0].plot(timePoints, lgobjectArea, color=lineColor[n,:], linewidth=0.8)
    axs[0].set_xlabel('Time (hrs)', fontsize=Label_Font, fontweight='bold')
    axs[0].set_ylabel('log2[Area]', fontsize=Label_Font, fontweight='bold')
    axs[0].set_title(roiID, fontsize=Label_Font, fontweight='bold')

    

    for n in range(1,5):
        axs.append(plt.axes(axesPos[n,:], xlim=xLim[n,:], ylim=yLim[n,:]))
        axs[n].plot(nbins[n,:],normVirts[n,:],color=lineColor[n,:])
        xPos = 0.7*np.abs(np.diff(xLim[n,:]))+xLim[n,0]
        axs[n].text(xPos,0.75, textLbls[n], fontsize = Label_Font,fontweight='bold',color=lineColor[n,:])

    if saveAll:
        plt.savefig(savePath)
    else:
        plt.show()

    return None

def figExpSummary(expDict, organism='Yeast', title = ''):
    plt.close('all')

    Label_Font = 12
    Title_Font = 12

    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = Title_Font
    mpl.rcParams['axes.labelsize'] = Label_Font

    plotDict = setPlotRange(organism)
    
    rngGCs  = plotDict['GCs']['xRange']
    rngTdbl = plotDict['Dbl']['xRange']
    rngTlag = plotDict['Lag']['xRange']
    rngTexp = plotDict['Tex']['xRange']
    rngNDub = plotDict['NumDbl']['xRange']
    rngPopNum = plotDict['PopNum']['xRange']
    cntrLbl = ['Dbl', 'Lag', 'Tex', 'NumDbl', 'PopNum']
    tickList = {}

    left   = 1.25/6
    bottom = 0.4/10
    width  = 1.2/8
    height = 9/10
    spacing = 0.05/6

    xLim = np.array([rngTdbl,
                     rngTlag,
                     rngTexp,
                     rngNDub,
                     rngPopNum], dtype = 'float64')

    textLbls= ['Td (hrs)', 'Tlag (hrs)','Texp (hrs)','Num Dbl','Pop Cnt']

    Path = mpath.Path
    commands = {'M': (mpath.Path.MOVETO,),
                'L': (mpath.Path.LINETO,),
                'Q': (mpath.Path.CURVE3,)*2,
                'C': (mpath.Path.CURVE4,)*3,
                'Z': (mpath.Path.CLOSEPOLY,)}

    numbins   = 75
    fitCol  = [6,5,3,1]
    # breakpoint()
    devisor = [ 
        plotDict['Dbl']['devisor'],
        plotDict['Lag']['devisor'], 
        plotDict['Tex']['devisor'], 
        plotDict['NumDbl']['devisor']
        ]
    
    
    roiList = [*expDict.keys()]
    key1='roiInfo'
    key2='Strain ID'
    
    yTickLbl=[]
    for roi in  expDict.keys():
        if len(expDict[roi][key1])>0:

            yTickLbl.append(expDict[roi][key1][key2])
        else:
            yTickLbl.append(roi)

    roiList = [x for _, x in sorted( zip(yTickLbl, roiList), key=lambda pair: pair[0])]
    
    roiList.reverse()

    yTickLbl.sort()
    yTickLbl.reverse()
    yTickLbl.insert(0,'')
    yTickLbl.append('')

    numRoi = len(roiList)
    poptot    = np.zeros((numRoi+1,2),  dtype='int')
    wScale = 0.8
    pathDict = {}
    cntr=0
    for key in roiList:
        cntr+=1
        normVirts = np.zeros((5,numbins), dtype='float64')
        virts     = np.zeros((5,numbins), dtype='float64')
        nbins     = np.zeros((5,numbins), dtype='float64')
    
        fitData = expDict[key]['fitData']
        poptot[cntr,:] = fitData.shape
        pathDict[key]={}
        with np.errstate(divide='ignore',invalid='ignore'):
            for n in range(4):
                nbins[n,:] = np.linspace(xLim[n,0], xLim[n,1], num=numbins)
                virts[n,:] = histogram1d( fitData[:,fitCol[n]]/devisor[n], numbins, xLim[n,:], weights = None)
                normVirts[n,:] = (virts[n,:]/np.max(virts[n,2:-10]))*wScale

                codes, verts = parseVirts(nbins[n,:], normVirts[n,:])
                verts[:,1] += cntr-0.5
                path = mpath.Path(verts, codes)
                pathDict[key][textLbls[n]] = path

        pathDict[key]['nbins'] = nbins
        pathDict[key]['normVirts'] = normVirts
        
    axesPos = np.zeros((5,4),dtype = 'float')
    for n in range(5):
        axesPos[n,:] =  [left+n*(width+spacing),bottom,width,height]

    gcFig = plt.figure(figsize=[7,9], dpi=100, facecolor='w')
    gcFig.suptitle(title, x=0.01, y= 0.9875, fontsize=16, fontweight = 'bold', horizontalalignment = 'left')

    axs = []

    n = 0
    xTicks = plotDict[cntrLbl[n]]['xTicks']
    xticklabels = [str(value) for value in xTicks]

    axs.append(plt.axes(axesPos[n,:], xlim=xLim[n,:], ylim=[0,numRoi+1], yticks=list(range(numRoi+1)), xticks=xTicks))

    axs[n].set_yticklabels(yTickLbl, fontsize=6, fontweight = 'bold')
    axs[n].set_xticklabels(xticklabels, fontsize=8, fontweight = 'bold', rotation= 45 )
    axs[n].set_title(textLbls[n], fontsize=10, fontweight = 'bold' )
    
    for roi in roiList:
            patch = mpatches.PathPatch(pathDict[roi][textLbls[n]], facecolor = [0,0,1,1], edgecolor = None, linewidth = 0 )
            axs[n].add_patch(patch)

    for n in range(1,4):
        xTicks = plotDict[cntrLbl[n]]['xTicks']
        xticklabels = [str(value) for value in xTicks]
        axs.append(plt.axes(axesPos[n,:], xlim=xLim[n,:], ylim=[0,numRoi+1], yticks=[], xticks=xTicks))
        axs[n].set_xticklabels(xticklabels, fontsize=8, fontweight = 'bold', rotation= 45 )
        axs[n].set_title(textLbls[n], fontsize=10, fontweight = 'bold' )

        for roi in roiList:
            patch = mpatches.PathPatch(pathDict[roi][textLbls[n]], facecolor = [0,0,1,1], edgecolor = None, linewidth = 0 )
            axs[n].add_patch(patch)

    
    n +=1
    xTicks = plotDict[cntrLbl[n]]['xTicks']
    xticklabels = [str(value) for value in xTicks]
    ypos = np.arange(poptot.shape[0])
    xstart = np.zeros((poptot.shape[0],),dtype = 'float')
   
    axs.append(plt.axes(axesPos[n,:], xscale = 'log', xlim=[1,10000], ylim=[0,numRoi+1], yticks=[], xticks=xTicks))
    axs[n].hlines(ypos, xstart, poptot[:,0], linewidth = 5, color = [0,0,1,1] )
    axs[n].set_yticklabels(yTickLbl,    fontsize=6, fontweight = 'bold')
    axs[n].set_xticklabels(xticklabels, fontsize=8, fontweight = 'bold', rotation= 45 )
    axs[n].set_title(textLbls[n], fontsize=10, fontweight = 'bold' )
    
    plt.show()

    return None

def stitchIm( roiLbl, imNum, imageDir, dataDir):
    imagePath = pathlib.Path(imageDir)
    
    # Load Region of Interest Data.  This HDF5 file should containt location of image stitch coordinates 
    dataPath = pathlib.Path(dataDir)
    initPath = list(dataPath.glob('*Index_ODELAYData.hdf5'))
    initData = fio.loadData(initPath[0])
    background = initData['backgroundImage']

    roiFolder = pathlib.Path('./'+ roiLbl)
    imageFileName = [*initData['roiFiles'][roiLbl]]

    imageFileName.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    imageFilePath = imagePath / roiFolder / imageFileName[imNum] 

    pixSize = initData['pixSize']
    magnification = initData['magnification']
 

    anImage = opl.stitchImage(imageFilePath, pixSize, magnification, background)
    
    im = anImage['Bf']
    imSize = im.shape

    # This data should be recorded from image display to make sure the image is visible.
    imageHist = histogram1d(im.ravel(),2**16,[0,2**16],weights = None).astype('float')
            # Calculate the cumulative probability ignoring zero values 
    cumHist = np.cumsum(imageHist)
    cumProb = (cumHist-cumHist[0])/(cumHist[2**16-1]-cumHist[0])
            # set low and high values ot normalize image contrast.        
    loval = np.argmax(cumProb>0.00001)
    hival = np.argmax(cumProb>=0.9995)

    adjIm = np.array((im.astype('float') - loval.astype('float'))/(hival.astype('float') - loval.astype('float'))*254, dtype = 'uint8')

    rsIm = cv2.resize(adjIm, (round(imSize[1]/5), round(imSize[0]/5)))

    cv2.imshow('Display Image', rsIm)
    k = cv2.waitKey(0)

    if k == 107 or k == -1:
        cv2.destroyWindow('Display Image')

    return k

def showImage(roiLbl, imNum, imageDir, dataDir):

    # image = odp.stitchImage(imageFileName, pixSize, magnification, background)

    expPath = pathlib.Path(imageDir)
    
    # Generate image file Path by combining the region of interest lable with the experiment path
    roiFolder = pathlib.Path('./'+ roiLbl)
    imageFileName = pathlib.Path('./'+ roiLbl + '_'+ f'{imNum:00d}' + '.mat')
    imageFilePath = expPath / roiFolder / imageFileName 
    
    # Load Region of Interest Data.  This HDF5 file should containt location of image stitch coordinates 
    dataPath = pathlib.Path(dataDir)
    initPath = list(dataPath.glob('*Index_ODELAYData.hdf5'))
    initData = fio.loadData(initPath[0])
    roiPath = dataPath / 'ODELAY Roi Data' / f'{roiLbl}.hdf5'

    roiData = fio.loadData(roiPath)
    background = initData['backgroundImage']

    # This data should be extracted from the Experiment Index file or stage data file.
    pixSize = initData['pixSize']
    magnification = initData['magnification']

    stInd = f'{imNum-1:03d}'
    stitchCorners = roiData['stitchMeta'][stInd]['imPix']
    # breakpoint()
    anImage = opl.assembleImage(imageFilePath, pixSize, magnification, background, stitchCorners)
    im = anImage['Bf']
    # im = opl.SobelGradient(im)
    imSize = im.shape

    # This data should be recorded from image display to make sure the image is visible.
    imageHist = histogram1d(im.ravel(),2**16,[0,2**16],weights = None).astype('float')
    # Calculate the cumulative probability ignoring zero values 
    cumHist = np.cumsum(imageHist)
    cumProb = (cumHist-cumHist[0])/(cumHist[2**16-1]-cumHist[0])
            # set low and high values ot normalize image contrast.        
    loval = np.argmax(cumProb>0.00001)
    hival = np.argmax(cumProb>=0.9995)

    adjIm = np.array((im.astype('float') - loval.astype('float'))/(hival.astype('float') - loval.astype('float'))*254, dtype = 'uint8')

    rsIm = cv2.resize(adjIm, (round(imSize[1]/5), round(imSize[0]/5)))

    cv2.imshow('Display Image', rsIm)
    k = cv2.waitKey(0)

    if k == 107 or k == -1:
        cv2.destroyWindow('Display Image')

    return k

def setPlotRange(organism=None):


    plotRange = {}
    plotRange['Mtb'] = {}
    plotRange['Mtb']['GCs'] = {}
    plotRange['Mtb']['Dbl'] = {}
    plotRange['Mtb']['Lag'] = {}
    plotRange['Mtb']['Tex'] = {}
    plotRange['Mtb']['Area'] = {}
    plotRange['Mtb']['NumDbl'] = {}
    plotRange['Mtb']['PopNum'] = {}

    plotRange['Mtb']['GCs']['xRange'] = [0, 170]
    plotRange['Mtb']['GCs']['yRange'] = [4, 14]
    plotRange['Mtb']['GCs']['xTicks'] = np.arange(0,100,20)
    plotRange['Mtb']['GCs']['xLabel'] = 'Hours'
    plotRange['Mtb']['GCs']['titleFrag'] = 'Dbl Time Hr'
    plotRange['Mtb']['GCs']['devisor'] = 60
    plotRange['Mtb']['GCs']['numObservations'] = 20

    plotRange['Mtb']['Dbl']['xRange'] = [0, 100]
    plotRange['Mtb']['Dbl']['xTicks'] = [20,40,60,80]
    plotRange['Mtb']['Dbl']['xStep'] = 5
    plotRange['Mtb']['Dbl']['xLabel'] = 'Hours'
    plotRange['Mtb']['Dbl']['titleFrag'] = 'Dbl Time Hr'
    plotRange['Mtb']['Dbl']['devisor'] = 60

    plotRange['Mtb']['Lag']['xRange'] = [0, 100]
    plotRange['Mtb']['Lag']['xTicks'] = [20,40,60,80]
    plotRange['Mtb']['Lag']['xStep'] = 2
    plotRange['Mtb']['Lag']['xLabel'] = 'Hours'
    plotRange['Mtb']['Lag']['titleFrag'] = 'Lag Time Hr'
    plotRange['Mtb']['Lag']['devisor'] = 60
    
    plotRange['Mtb']['Tex']['xRange'] = [0, 100]
    plotRange['Mtb']['Tex']['xTicks'] = [20,40,60,80]
    plotRange['Mtb']['Tex']['xStep'] = 2
    plotRange['Mtb']['Tex']['xLabel'] = 'Hours'
    plotRange['Mtb']['Tex']['titleFrag'] = 'Tex Hr'
    plotRange['Mtb']['Tex']['devisor'] = 30

    plotRange['Mtb']['Area']['xRange'] = [0, 30]
    plotRange['Mtb']['Area']['xTicks'] = [2,4,6,8]
    plotRange['Mtb']['Area']['xStep'] = 0.25
    plotRange['Mtb']['Area']['xLabel'] = 'log2 Pixels'
    plotRange['Mtb']['Area']['titleFrag'] = 'log2 Area'
    plotRange['Mtb']['Area']['devisor'] = 1

    plotRange['Mtb']['NumDbl']['xRange'] = [0, 10]
    plotRange['Mtb']['NumDbl']['xTicks'] = [2,4,6,8]
    plotRange['Mtb']['NumDbl']['xStep'] = 0.25
    plotRange['Mtb']['NumDbl']['xLabel']    = 'Num Dbl Rel'
    plotRange['Mtb']['NumDbl']['titleFrag'] = 'Num Dbl Rel'
    plotRange['Mtb']['NumDbl']['devisor'] = 1

    plotRange['Mtb']['PopNum']['xRange'] = [0, 10000]
    plotRange['Mtb']['PopNum']['xTicks'] = [10,100,1000]
    plotRange['Mtb']['PopNum']['xStep'] = 10
    plotRange['Mtb']['PopNum']['xLabel']  = 'log10 Pop'
    plotRange['Mtb']['PopNum']['titleFrag'] = 'Pop Num'
    plotRange['Mtb']['PopNum']['devisor'] = 1

    plotRange['Mabs'] = {}
    plotRange['Mabs']['GCs'] = {}
    plotRange['Mabs']['Dbl'] = {}
    plotRange['Mabs']['Lag'] = {}
    plotRange['Mabs']['Tex'] = {}
    plotRange['Mabs']['Area'] = {}
    plotRange['Mabs']['NumDbl'] = {}
    plotRange['Mabs']['PopNum'] = {}

    plotRange['Mabs']['GCs']['xRange'] = [0, 70]
    plotRange['Mabs']['GCs']['yRange'] = [4, 16]
    plotRange['Mabs']['GCs']['xTicks'] = np.arange(0,70,10)
    plotRange['Mabs']['GCs']['xLabel'] = 'Hours'
    plotRange['Mabs']['GCs']['titleFrag'] = 'Dbl Time Hr'
    plotRange['Mabs']['GCs']['devisor'] = 60
    plotRange['Mabs']['GCs']['numObservations'] = 20

    plotRange['Mabs']['Dbl']['xRange'] = [0, 10]
    plotRange['Mabs']['Dbl']['xTicks'] = [2,4,6,8]
    plotRange['Mabs']['Dbl']['xStep'] = 0.5
    plotRange['Mabs']['Dbl']['xLabel'] = 'Hours'
    plotRange['Mabs']['Dbl']['titleFrag'] = 'Dbl Time Hr'
    plotRange['Mabs']['Dbl']['devisor'] =  60

    plotRange['Mabs']['Lag']['xRange'] = [0, 40]
    plotRange['Mabs']['Lag']['xTicks'] = [10,20,30]
    plotRange['Mabs']['Lag']['xStep'] = 1
    plotRange['Mabs']['Lag']['xLabel'] = 'Hours'
    plotRange['Mabs']['Lag']['titleFrag'] = 'Lag Time Hr'
    plotRange['Mabs']['Lag']['devisor'] =  60

    plotRange['Mabs']['Tex']['xRange'] = [0, 40]
    plotRange['Mabs']['Tex']['xTicks'] = [10,20,30]
    plotRange['Mabs']['Tex']['xStep'] = 1
    plotRange['Mabs']['Tex']['xLabel'] = 'Hours'
    plotRange['Mabs']['Tex']['titleFrag'] = 'Tex Hr'
    plotRange['Mabs']['Tex']['devisor'] =   30

    plotRange['Mabs']['Area']['xRange'] = [0, 30]
    plotRange['Mabs']['Area']['xTicks'] = [20,40,60,80]
    plotRange['Mabs']['Area']['xStep'] = 0.25
    plotRange['Mabs']['Area']['xLabel'] = 'log2 Pixels'
    plotRange['Mabs']['Area']['titleFrag'] = 'log2 Area'
    plotRange['Mabs']['Area']['devisor'] =  1

    plotRange['Mabs']['NumDbl']['xRange'] = [0, 10]
    plotRange['Mabs']['NumDbl']['xTicks'] = [2,4,6,8]
    plotRange['Mabs']['NumDbl']['xStep'] = 0.25
    plotRange['Mabs']['NumDbl']['xLabel']  = 'log2 Pixels'
    plotRange['Mabs']['NumDbl']['titleFrag'] = 'Num Dbl Rel'
    plotRange['Mabs']['NumDbl']['devisor'] =  1

    plotRange['Mabs']['PopNum']['xRange'] = [0, 10000]
    plotRange['Mabs']['PopNum']['xTicks'] = [10,100,1000]
    plotRange['Mabs']['PopNum']['xStep'] = 10
    plotRange['Mabs']['PopNum']['xLabel']  = 'log10 Pop'
    plotRange['Mabs']['PopNum']['titleFrag'] = 'Pop Num'
    plotRange['Mabs']['PopNum']['devisor'] =  1

    plotRange['Yeast'] = {}
    plotRange['Yeast']['GCs'] = {}
    plotRange['Yeast']['Dbl'] = {}
    plotRange['Yeast']['Lag'] = {}
    plotRange['Yeast']['Tex'] = {}
    plotRange['Yeast']['Area'] = {}
    plotRange['Yeast']['NumDbl'] = {}
    plotRange['Yeast']['PopNum'] = {}

    plotRange['Yeast']['GCs']['xRange'] = [0, 3000]
    plotRange['Yeast']['GCs']['yRange'] = [4, 16]
    plotRange['Yeast']['GCs']['xTicks'] = [100,200,300,400]
    plotRange['Yeast']['GCs']['xStep'] = 4
    plotRange['Yeast']['GCs']['xLabel'] = 'Minutes'
    plotRange['Yeast']['GCs']['titleFrag'] = 'Time Min'
    plotRange['Yeast']['GCs']['devisor'] =  1
    plotRange['Yeast']['GCs']['numObservations'] = 10

    plotRange['Yeast']['Dbl']['xRange'] = [25, 500]
    plotRange['Yeast']['Dbl']['xTicks'] = [100,200,300,400,500]
    plotRange['Yeast']['Dbl']['xStep'] = 4
    plotRange['Yeast']['Dbl']['xLabel'] = 'Minutes'
    plotRange['Yeast']['Dbl']['titleFrag'] = 'Dbl Time Min'
    plotRange['Yeast']['Dbl']['devisor'] =  1

    plotRange['Yeast']['Lag']['xRange'] = [0, 3000]
    plotRange['Yeast']['Lag']['xTicks'] = [500,1000,1500,2000, 2500]
    plotRange['Yeast']['Lag']['xStep'] = 1
    plotRange['Yeast']['Lag']['xLabel'] = 'Minutes'
    plotRange['Yeast']['Lag']['titleFrag'] = 'Lag Time Min'
    plotRange['Yeast']['Lag']['devisor'] =  1

    plotRange['Yeast']['Tex']['xRange'] = [0, 3000]
    plotRange['Yeast']['Tex']['xTicks'] = [500,1000,1500,2000, 2500]
    plotRange['Yeast']['Tex']['xStep'] = 1
    plotRange['Yeast']['Tex']['xLabel'] = 'Minutes'
    plotRange['Yeast']['Tex']['titleFrag'] = 'Tex Min'
    plotRange['Yeast']['Tex']['devisor'] =  0.5

    plotRange['Yeast']['Area']['xRange'] = [0, 40]
    plotRange['Yeast']['Area']['xTicks'] = [10,20,30]
    plotRange['Yeast']['Area']['xStep'] = 0.5
    plotRange['Yeast']['Area']['xLabel'] = 'log2 Pixels'
    plotRange['Yeast']['Area']['titleFrag'] = 'log2 Area'
    plotRange['Yeast']['Area']['devisor'] =  1

    plotRange['Yeast']['NumDbl']['xRange'] = [0, 20]
    plotRange['Yeast']['NumDbl']['xTicks'] = [2,4,6,8]
    plotRange['Yeast']['NumDbl']['xStep'] = 0.25
    plotRange['Yeast']['NumDbl']['xLabel']  = 'log2 Pixels'
    plotRange['Yeast']['NumDbl']['titleFrag'] = 'Num Dbl Rel'
    plotRange['Yeast']['NumDbl']['devisor'] =  1

    plotRange['Yeast']['PopNum']['xRange'] = [0, 10000]
    plotRange['Yeast']['PopNum']['xTicks'] = [10,100,1000]
    plotRange['Yeast']['PopNum']['xStep'] = 10
    plotRange['Yeast']['PopNum']['xLabel']  = 'log10 Pop'
    plotRange['Yeast']['PopNum']['titleFrag'] = 'Pop Num'
    plotRange['Yeast']['PopNum']['devisor'] =  1

    if organism == None:
        return plotRange
    else:
        return plotRange[organism]

def scaleImage(im, lowcut = 0.00001, highcut = 0.9995, scaleImage = 1, intensityScale = 254, dtypeOut = 'uint8'):

    # make a histogram of the image in the bitdept that the image was recorded.  
    imageHist = histogram1d(im.ravel(),2**16,[0,2**16],weights = None).astype('float')


    # Calculate the cumulative probability ignoring zero values 
    cumHist = np.empty(imageHist.shape, dtype='float')
    cumHist[0]  = 0
    cumHist[1:] = np.cumsum(imageHist[1:])

    # if you expect a lot of zero set 
    cumRange = cumHist[2**16-1]-cumHist[0]
    # if you expect a lot of zero set 
    cumHist-=cumHist[0]
    cumHist /=cumRange

    # set low and high values ot normalize image contrast.  
    loval = np.argmax(cumHist>=lowcut)
    hival = np.argmax(cumHist>=highcut)
    scIm = np.clip(im, loval, hival).astype('float')
    # scale the image linearly over the range given.  This does not set alpha values or whatever.
    scaleFactor = intensityScale/(hival-loval)
    scIm -=loval
    scIm *= scaleFactor
    adjIm = np.require(scIm, dtype = dtypeOut, requirements = 'C')

    # resize if you need to 
    rsIm = cv2.resize(adjIm, (round(im.shape[1]/scaleImage), round(im.shape[0]/scaleImage)))

    
    return rsIm

def parseVirts(x, y):

    commands = {'M': (mpath.Path.MOVETO,),
                'L': (mpath.Path.LINETO,),
                'Q': (mpath.Path.CURVE3,)*2,
                'C': (mpath.Path.CURVE4,)*3,
                'Z': (mpath.Path.CLOSEPOLY,)}

    rc = y.shape
    vertices = np.zeros((rc[0]+3,2),dtype='float')
    vertices[0,:] = [x[0],y[0]]
    codes = []
    codes.extend(commands['M'])
    for n in range(1,rc[0]):
        codes.extend(commands['L'])
        
        vertices[n,:] = [x[n],y[n]] 


    vertices[-3,:] = [x[-1],0]
    codes.extend(commands['L'])
    vertices[-2,:] = [0,0]
    codes.extend(commands['L'])
    vertices[-2,:] = [0,0]
    codes.extend(commands['Z'])
    return codes, vertices

class OImageView(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(OImageView, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.qImage = QImage()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)


    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None, reset=True):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        if reset:
            self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(OImageView, self).mousePressEvent(event)
# Window is called to view window.  
class ImageWindow(QtWidgets.QWidget):
    '''
    ImageWindow:  a QWidget which holds the GraphicsView and button elements
    '''
    def __init__(self):
        super(ImageWindow, self).__init__()

        # Load experiment and odelayConfig data into Window data.
        self.odelayConfig   = fio.loadConfig()
        self.experimentData = self.loadExperimentData()
        self.roiList = [*self.experimentData['roiFiles']]
        self.roiLbl = self.roiList[0]
        self.numImages=len(self.experimentData['roiFiles'][self.roiLbl])
        self.imageNumber = 1

        #Create Photoviewer object
        self.viewer = OImageView(self)

        # 'Load image' button
        self.selectRoi = QtWidgets.QComboBox(self)
        qroiList = [self.tr(item) for item in self.roiList]
        self.selectRoi.addItems(qroiList)
        self.selectRoi.currentTextChanged.connect(self.chooseRoi)

        #Button for load previous Image
        self.btnPrevImage = QtWidgets.QToolButton(self)
        self.btnPrevImage.setText('Prev')
        self.btnPrevImage.setObjectName('btnPrevImage')
        self.btnPrevImage.clicked.connect(self.changeImage)

        #Button for load previous Image
        self.btnNextImage = QtWidgets.QToolButton(self)
        self.btnNextImage.setText('Next')
        self.btnNextImage.setObjectName('btnNextImage')
        self.btnNextImage.clicked.connect(self.changeImage)

        #Button for load previous Image
        self.btnSaveImage = QtWidgets.QToolButton(self)
        self.btnSaveImage.setText('Save')
        self.btnSaveImage.setObjectName('btnSaveImage')
        self.btnSaveImage.clicked.connect(self.saveImage)

        # Button to change from drag/pan to getting pixel info
        self.btnPixInfo = QtWidgets.QToolButton(self)
        self.btnPixInfo.setText('Enter pixel info mode')
        self.btnPixInfo.clicked.connect(self.pixInfo)

        self.editPixInfo = QtWidgets.QLineEdit(self)
        self.editPixInfo.setReadOnly(True)
        self.viewer.photoClicked.connect(self.photoClicked)
        
        # Add Image time slider
        self.imageSlider = QSlider(Qt.Horizontal)       
        self.imageSlider.setRange(1,self.numImages)
        self.imageSlider.sliderReleased.connect(self.changeImage)

        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        VBlayout.addWidget(self.imageSlider)

        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.selectRoi)
        HBlayout.addWidget(self.btnPrevImage)
        HBlayout.addWidget(self.btnNextImage)
        HBlayout.addWidget(self.btnSaveImage)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)

    def chooseRoi(self, ind):

        self.roiLbl = ind
        self.numImages = len(self.experimentData['roiFiles'][self.roiLbl])
        if self.imageNumber>self.numImages:
            self.imageNumber = self.numImages
            self.imageSlider.setValue = self.numImages
        
        self.loadImage()

    def loadImage(self):
    
        self.viewer.qImage = self.readImage()
        pixmap = QPixmap.fromImage(self.viewer.qImage)
        self.viewer.setPhoto(pixmap)

    def saveImage(self):
        location = self.odelayConfig['LocalDataDir']
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Image", self.tr(location),"Images (*.png, *.jpg)", options=options)
        print(fileName)
        val = self.viewer.qImage.save(fileName, format=None, quality=100)
        if val:
            print('Image saved')

    def changeImage(self):

        sending_widget = self.sender()

        if sending_widget.objectName() == self.btnNextImage.objectName():
            self.imageNumber += 1
            if self.imageNumber>self.numImages:
                self.imageNumber = self.numImages
            else:
                self.viewer.qImage = self.readImage()
                pixmap = QPixmap.fromImage(self.viewer.qImage)
                self.imageSlider.setValue(self.imageNumber)
                self.viewer.setPhoto(pixmap, False)

        elif sending_widget.objectName() == self.btnPrevImage.objectName():
            self.imageNumber -= 1
            if self.imageNumber<1:
                self.imageNumber = 1
            else:
                self.viewer.qImage = self.readImage()
                pixmap = QPixmap.fromImage(self.viewer.qImage)
                self.imageSlider.setValue(self.imageNumber)
                self.viewer.setPhoto(pixmap, False)

        elif sending_widget.objectName() == self.imageSlider.objectName():
            self.imageNumber = sending_widget.value()
            self.viewer.qImage = self.readImage()
            pixmap = QPixmap.fromImage(self.viewer.qImage)
            self.viewer.setPhoto(pixmap, False)

    def pixInfo(self):
        self.viewer.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def openFileDialog():
    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None,"Select ODELAY Data Set", "","ODELAYExpDisc (*Index_ODELAYData.mat);; Mat-Files (*.mat)", options=options)
        return fileName

    def loadExperimentData(self):

        imagePath = pathlib.Path(self.odelayConfig['LocalImageDir'])
        dataPath  = pathlib.Path(self.odelayConfig['LocalDataDir'])
        indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]

        if len(indexList)==1:
            expIndexPath = dataPath / indexList[0]
            expData = fio.loadData(expIndexPath)

        return expData

    def readImage(self, lowcut = 0.0005, highcut = 0.99995):

        roiLbl = self.roiLbl
        imNum  = self.imageNumber

        imagePath = pathlib.Path(self.odelayConfig['LocalImageDir'])
        dataPath  = pathlib.Path(self.odelayConfig['LocalDataDir'])
        # Generate image file Path by combining the region of interest lable with the experiment path
        roiFolder = pathlib.Path('./'+ roiLbl)
        imageFileName = pathlib.Path('./'+ roiLbl + '_'+ f'{imNum:00d}' + '.mat')
        imageFilePath = imagePath / roiFolder / imageFileName 
        
        # Load Region of Interest Data.  This HDF5 file should containt location of image stitch coordinates 
        roiPath = dataPath / 'ODELAY Roi Data' / f'{roiLbl}.hdf5'

        roiData    = fio.loadData(roiPath)
        background = self.experimentData['backgroundImage']

        # This data should be extracted from the Experiment Index file or stage data file.
        pixSize       = self.experimentData['pixSize']
        magnification = self.experimentData['magnification']

        stInd = f'{imNum-1:03d}'
        stitchCorners = roiData['stitchMeta'][stInd]['imPix']
   
        anImage = opl.assembleImage(imageFilePath, pixSize, magnification, background, stitchCorners)
        im = anImage['Bf']
        # make a histogram of the image in the bitdept that the image was recorded.  
        imageHist = histogram1d(im.ravel(),2**16,[0,2**16],weights = None).astype('float')

        # Calculate the cumulative probability ignoring zero values 
        cumHist = np.zeros(imageHist.shape, dtype='float')
        cumHist[1:] = np.cumsum(imageHist[1:])

        # if you expect a lot of zero set 
        cumProb = (cumHist-cumHist[0])/(cumHist[2**16-1]-cumHist[0])

        # set low and high values ot normalize image contrast.  
        loval = np.argmax(cumProb>=lowcut)
        hival = np.argmax(cumProb>=highcut)

        scIm = (im.astype('float') - loval.astype('float'))/(hival.astype('float') - loval.astype('float'))*254
        lim = np.iinfo('uint8')
        scIm = np.clip(scIm, lim.min, lim.max)
        # Set image data type and make sure the array is contiguous in memory.  
        imageData = np.require(scIm, dtype = 'uint8', requirements = 'C')  
        # Set data as a QImage.  This is a greyscale image 
        Qim = QImage(imageData.data, imageData.shape[1], imageData.shape[0], imageData.shape[1], QImage.Format_Grayscale8)
                    
        Qim.data = imageData
        
        return Qim

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("PyQt Video Player Widget Example - pythonprogramminglanguage.com") 

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderReleased.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)        
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def openFile(self):

        odelayConfig = fio.loadConfig()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                        odelayConfig['LocalDataDir'])

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self):
        position = self.positionSlider.value()
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

def videoViewer():

    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())

def imageViewer():
    
    app = QtWidgets.QApplication(sys.argv)
    window = ImageWindow()
    window.setGeometry(500, 300, 800, 600)
    window.show()
    window.loadImage()
    sys.exit(app.exec_())

def waveLengthToRGB(wl=650):
    try:
        wl=int(wl)
    except:
        wl=450
    # print(wl)
    if wl<380:
        wl= 380
    elif wl>780:
        wl = 780

    if wl>=380 and wl<=440:
        R = np.abs((wl-440)/(440-380))
        G = 0
        B = 1

    elif wl>440 and wl<=490:
        R = 0
        G = np.abs((wl-440)/(490-440))
        B = 1

    elif wl>490 and wl<=510:
        R = 0
        G = 1
        B = np.abs((wl-510)/(510-490))

    elif wl>510 and wl<=580:
        R = np.abs((wl-510)/(580-510))
        G = 1
        B = 0;

    elif wl>580 and wl<=645:
        R = 1;
        G = np.abs((wl-645)/(645-580))
        B = 0

    elif wl>645 and wl<=780:
        R = 1
        G = 0
        B = 0

    # LET THE INTENSITY SSS FALL OFF NEAR THE VISION LIMITS
    if wl>700:
        SSS=0.3+0.7* (780-wl)/(780-700)
    elif wl<420:
        SSS=.3+.7*(wl-380)/(420-380)
    else:
        SSS=1

    r = np.round(SSS*R*255).astype('uint8')
    g = np.round(SSS*G*255).astype('uint8')
    b = np.round(SSS*B*255).astype('uint8')
    return [r,g,b]

# class FocusPlot(QMainWindow):
#     def __init__(self, parent=None):
#         QMainWindow.__init__(self, parent)
#         self.setWindowTitle('Demo: PyQt with matplotlib')

#         self.create_menu()
#         self.create_main_frame()
#         self.create_status_bar()

#         self.textbox.setText('1 2 3 4')
#         self.on_draw()

#     def save_plot(self):
#         file_choices = "PNG (*.png)|*.png"
        
#         path, ext = QFileDialog.getSaveFileName(self, 
#                         'Save file', '', 
#                         file_choices)
#         path = path.encode('utf-8')
#         if not path[-4:] == file_choices[-4:].encode('utf-8'):
#             path += file_choices[-4:].encode('utf-8')
#         print(path)
#         if path:
#             self.canvas.print_figure(path.decode(), dpi=self.dpi)
#             self.statusBar().showMessage('Saved to %s' % path, 2000)
    
#     def on_about(self):
#         msg = """ A demo of using PyQt with matplotlib:
        
#          * Use the matplotlib navigation bar
#          * Add values to the text box and press Enter (or click "Draw")
#          * Show or hide the grid
#          * Drag the slider to modify the width of the bars
#          * Save the plot to a file using the File menu
#          * Click on a bar to receive an informative message
#         """
#         QMessageBox.about(self, "About the demo", msg.strip())
    
#     def on_pick(self, event):
#         # The event received here is of the type
#         # matplotlib.backend_bases.PickEvent
#         #
#         # It carries lots of information, of which we're using
#         # only a small amount here.
#         # 
#         box_points = event.artist.get_bbox().get_points()
#         msg = "You've clicked on a bar with coords:\n %s" % box_points
        
#         QMessageBox.information(self, "Click!", msg)
    
#     def on_draw(self):
#         """ Redraws the figure
#         """
#         str = self.textbox.text().encode('utf-8')
#         self.data = [int(s) for s in str.split()]
        
#         x = range(len(self.data))

#         # clear the axes and redraw the plot anew
#         #
#         self.axes.clear()        
#         self.axes.grid(self.grid_cb.isChecked())
        
#         self.axes.bar(
#             x=x, 
#             height=self.data, 
#             width=self.slider.value() / 100.0, 
#             align='center', 
#             alpha=0.44,
#             picker=5)
        
#         self.canvas.draw()
    
#     def create_main_frame(self):
#         self.main_frame = QWidget()
        
#         # Create the mpl Figure and FigCanvas objects. 
#         # 5x4 inches, 100 dots-per-inch
#         #
#         self.dpi = 100
#         self.fig = Figure((5.0, 4.0), dpi=self.dpi)
#         self.canvas = FigureCanvas(self.fig)
#         self.canvas.setParent(self.main_frame)
        
#         # Since we have only one plot, we can use add_axes 
#         # instead of add_subplot, but then the subplot
#         # configuration tool in the navigation toolbar wouldn't
#         # work.
#         #
#         self.axes = self.fig.add_subplot(111)
        
#         # Bind the 'pick' event for clicking on one of the bars
#         #
#         self.canvas.mpl_connect('pick_event', self.on_pick)
        
#         # Create the navigation toolbar, tied to the canvas
#         #
#         self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
#         # Other GUI controls
#         # 
#         self.textbox = QLineEdit()
#         self.textbox.setMinimumWidth(200)
#         self.textbox.editingFinished.connect(self.on_draw)
        
#         self.draw_button = QPushButton("&Draw")
#         self.draw_button.clicked.connect(self.on_draw)
        
#         self.grid_cb = QCheckBox("Show &Grid")
#         self.grid_cb.setChecked(False)
#         self.grid_cb.stateChanged.connect(self.on_draw)
        
#         slider_label = QLabel('Bar width (%):')
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setRange(1, 100)
#         self.slider.setValue(20)
#         self.slider.setTracking(True)
#         self.slider.setTickPosition(QSlider.TicksBothSides)
#         self.slider.valueChanged.connect(self.on_draw)
        
#         #
#         # Layout with box sizers
#         # 
#         hbox = QHBoxLayout()
        
#         for w in [  self.textbox, self.draw_button, self.grid_cb,
#                     slider_label, self.slider]:
#             hbox.addWidget(w)
#             hbox.setAlignment(w, Qt.AlignVCenter)
        
#         vbox = QVBoxLayout()
#         vbox.addWidget(self.canvas)
#         vbox.addWidget(self.mpl_toolbar)
#         vbox.addLayout(hbox)
        
#         self.main_frame.setLayout(vbox)
#         self.setCentralWidget(self.main_frame)
    
#     def create_status_bar(self):
#         self.status_text = QLabel("This is a demo")
#         self.statusBar().addWidget(self.status_text, 1)
        
#     def create_menu(self):        
#         self.file_menu = self.menuBar().addMenu("&File")
        
#         load_file_action = self.create_action("&Save plot",
#             shortcut="Ctrl+S", slot=self.save_plot, 
#             tip="Save the plot")
#         quit_action = self.create_action("&Quit", slot=self.close, 
#             shortcut="Ctrl+Q", tip="Close the application")
        
#         self.add_actions(self.file_menu, 
#             (load_file_action, None, quit_action))
        
#         self.help_menu = self.menuBar().addMenu("&Help")
#         about_action = self.create_action("&About", 
#             shortcut='F1', slot=self.on_about, 
#             tip='About the demo')
        
#         self.add_actions(self.help_menu, (about_action,))

#     def add_actions(self, target, actions):
#         for action in actions:
#             if action is None:
#                 target.addSeparator()
#             else:
#                 target.addAction(action)

#     def create_action(  self, text, slot=None, shortcut=None, 
#                         icon=None, tip=None, checkable=False):
#         action = QAction(text, self)
#         if icon is not None:
#             action.setIcon(QIcon(":/%s.png" % icon))
#         if shortcut is not None:
#             action.setShortcut(shortcut)
#         if tip is not None:
#             action.setToolTip(tip)
#             action.setStatusTip(tip)
#         if slot is not None:
#             action.triggered.connect(slot)
#         if checkable:
#             action.setCheckable(True)
#         return action

# # def main():
# #     app = QApplication(sys.argv)
# #     form = AppForm()
# #     form.show()
# #     app.exec_()


# # if __name__ == "__main__":
# #     main()


# class InteractiveGCPlot(QWidget)