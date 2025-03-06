
import re
import os
import sys
import getpass
import pathlib
import time
import subprocess
import json
import asyncio
import functools

from datetime import datetime

import click
from fast_histogram import histogram1d
import paramiko
import jinja2
import numpy as np
import cv2
import tools.fileio     as fio
import tools.imagepl    as opl
import tools.odelayplot as odp
import tools.experimentModules as expMods
import odelaySetConfig

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
mpl.use('QT5Agg')


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5              import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets    import QApplication, QCheckBox, QComboBox, QDialog, QMainWindow, QGraphicsObject, QGraphicsPixmapItem
from PyQt5.QtWidgets    import QSizePolicy, QWidget, QInputDialog, QFileDialog
from PyQt5.QtWidgets    import QGraphicsView, QGraphicsScene, QGridLayout, QGroupBox, QHBoxLayout, QFrame
from PyQt5.QtWidgets    import QLabel, QPushButton, QStyle, QVBoxLayout, QWidget, QPushButton, QAction,  QGroupBox
from PyQt5.QtWidgets    import QLineEdit, QMainWindow,QMenu, QMenuBar,  QPushButton, QSlider, QSpinBox, QTextEdit, QWidget, QVBoxLayout
from PyQt5.QtGui        import QBrush, QColor, QImage, QPainter, QPen, QPixmap, qRgb, QTransform, QLinearGradient

from PyQt5.QtGui        import QImage, QPixmap, QIcon, QPainter, QColor, QBrush, QPainterPath, QPen, QLinearGradient
from PyQt5.QtCore       import QDir, Qt, QUrl, QPointF
from PyQt5.QtCore       import pyqtSignal, QMutex, QMutexLocker, QObject, QPoint, QSize, Qt,  QRectF, QTimer, QThread, QWaitCondition
from PyQt5.QtChart      import QChart, QChartView, QValueAxis, QBarCategoryAxis, QBarSet, QBarSeries


# TODO:   Nav Plate layout and imaging.  
#        1. Then rectangles to represent 

class OdelayStitchedImageViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(OdelayStitchedImageViewer, self).__init__(parent)
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
        super(OdelayStitchedImageViewer, self).mousePressEvent(event)

class TrackPosition(QtWidgets.QGraphicsObject):
    '''
    Graphics Object that highlights where the location of current tracks.
    '''
    trackSelected = pyqtSignal(int)
    def __init__(self, parent, trackBox= np.array((10,10,10,10), dtype='int'), trackIndex = 0):
        super(TrackPosition, self).__init__()
        self.color    = QColor(255,0,0,10)
        self.penColor = QColor(255,0,0,255)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.trackBox   = trackBox
        self.activePos  = False
        self.trackIndex = trackIndex
        
    def boundingRect(self):
        return QRectF(-self.trackBox[2]/2, -self.trackBox[3]/2, self.trackBox[2], self.trackBox[3])
    
    def paint(self, painter, option, widget):
        painter.setPen(QPen(self.penColor, 2))
        painter.setBrush(QBrush(self.color))
        painter.drawRect(-self.trackBox[2]/2, -self.trackBox[3]/2, self.trackBox[2], self.trackBox[3])
        
    def mousePressEvent(self,event):
        self.trackSelected.emit(self.trackIndex)
        super(TrackPosition, self).mousePressEvent(event)

class MplCanvas(FigureCanvasQTAgg):
    gcLinePicked = pyqtSignal(int)
    def __init__(self, parent=None, width=5, height=4, dpi=200, dataDict = {} ):
        
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(fig)
        
        # Extract Data for plotting
        xData     = dataDict['xData']
        yData     = dataDict['yData']
        axesPos   = dataDict['axesPos']
        lineColor = dataDict['lineColor']
        lineWidth = dataDict['lineWidth']
        
        self.lineColor = lineColor
        self.fig = fig
        self.axes = plt.axes(axesPos)
        self.axes.set_xlim(dataDict['xLim'])
        self.axes.set_ylim(dataDict['yLim'])

        if dataDict['drawRect']:
            xy = (0,0)
            width = xData[-1]
            height = self.axes.get_ylim()[-1]
            faceColor = lineColor.copy()
            edgeColor = faceColor
            faceColor[3] = 0
            self.rect = mpatches.Rectangle(xy, width, height, angle = 0.0, zorder = 2, facecolor = faceColor, edgecolor = edgeColor)
            self.axes.add_patch(self.rect)
            self.xyInit = xy
            self.xRange = dataDict['xLim']
            self.lock = True

            self.axes.plot(xData, yData, color=lineColor, linewidth=lineWidth, zorder = 3)

        else:
            self.gcLineDir = {}
            for n in range(yData.shape[1]):
                objID = dataDict['objID'][n]
                self.gcLineDir[str(objID)] = self.axes.plot(xData, yData[:,n], color=lineColor, linewidth=lineWidth, zorder = 3, gid = str(objID))[0]

        return None

class InteractiveGCs(QWidget):
    updateReady = pyqtSignal(str)  # send signal to trackobject to change color
    def __init__(self, plotData):
        super(InteractiveGCs, self).__init__()

        self.inds = []

        self.clickTimer  = QTimer()
        self.clickTimer.setSingleShot(True)
        self.clickTimer.setInterval(100)
        self.clickTimer.timeout.connect(self.sendPickedList)
        
        self.redraw = False
        self.wellChanged = False
        self.canvas = {}

        self.canvas['gcLines'] = MplCanvas(self, width=5, height=1, dpi=100, dataDict = plotData['gcLines'])
        self.canvas['dblHist'] = MplCanvas(self, width=5, height=1, dpi=100, dataDict = plotData['dblHist'])
        self.canvas['lagHist'] = MplCanvas(self, width=5, height=1, dpi=100, dataDict = plotData['lagHist'])
        self.canvas['texHist'] = MplCanvas(self, width=5, height=1, dpi=100, dataDict = plotData['texHist'])
        self.canvas['NuDHist'] = MplCanvas(self, width=5, height=1, dpi=100, dataDict = plotData['NuDHist'])

        virtLayout = QVBoxLayout()
        virtLayout.addWidget(self.canvas['gcLines'])
        virtLayout.addWidget(self.canvas['dblHist'])
        virtLayout.addWidget(self.canvas['lagHist'])
        virtLayout.addWidget(self.canvas['texHist'])
        virtLayout.addWidget(self.canvas['NuDHist'])
        virtLayout.setStretch(0,30)
        virtLayout.setStretch(1,10)
        virtLayout.setStretch(2,10)
        virtLayout.setStretch(3,10)
        virtLayout.setStretch(4,10)

        virtLayout.setSpacing(0)
        self.setLayout(virtLayout)

        self.canvas['gcLines'].mpl_connect("pick_event", self.gcPicked)

        self.canvas['dblHist'].mpl_connect("button_press_event",   self.dbl_on_press)
        self.canvas['dblHist'].mpl_connect("button_release_event", self.dbl_on_release)
        self.canvas['dblHist'].mpl_connect("motion_notify_event",  self.dbl_on_move)

        self.canvas['lagHist'].mpl_connect("button_press_event",   self.lag_on_press)
        self.canvas['lagHist'].mpl_connect("button_release_event", self.lag_on_release)
        self.canvas['lagHist'].mpl_connect("motion_notify_event",  self.lag_on_move)

        self.canvas['texHist'].mpl_connect("button_press_event",   self.tex_on_press)
        self.canvas['texHist'].mpl_connect("button_release_event", self.tex_on_release)
        self.canvas['texHist'].mpl_connect("motion_notify_event",  self.tex_on_move)

        self.canvas['NuDHist'].mpl_connect("button_press_event",   self.NuD_on_press)
        self.canvas['NuDHist'].mpl_connect("button_release_event", self.NuD_on_release)
        self.canvas['NuDHist'].mpl_connect("motion_notify_event",  self.NuD_on_move)
        
    def dbl_on_press(self, event):   
        self.on_press(event, 'dblHist')
    def lag_on_press(self, event):   
        self.on_press(event, 'lagHist')
    def tex_on_press(self, event):   
        self.on_press(event, 'texHist')
    def NuD_on_press(self, event):   
        self.on_press(event, 'NuDHist')

    def dbl_on_move(self, event):    
        self.on_move(event, 'dblHist')
    def lag_on_move(self, event):    
        self.on_move(event, 'lagHist')
    def tex_on_move(self, event):    
        self.on_move(event, 'texHist')
    def NuD_on_move(self, event):    
        self.on_move(event, 'NuDHist')
    
    def dbl_on_release(self, event): 
        self.on_release(event, 'dblHist')
    def lag_on_release(self, event): 
        self.on_release(event, 'lagHist')
    def tex_on_release(self, event): 
        self.on_release(event, 'texHist')
    def NuD_on_release(self, event): 
        self.on_release(event, 'NuDHist')

    def gcPicked(self, event):
        if not self.clickTimer.isActive():
            self.pickedList = []
            self.pickedArtist = []
            self.clickTimer.start()
            for ind in event.ind:
                self.pickedList.append(event.artist.get_gid())

        elif self.clickTimer.isActive():
            for ind in event.ind:
                self.pickedList.append(event.artist.get_gid())
        
        return None

    def sendPickedList(self):
        self.updateReady.emit('pickedInds')
        return None

    def on_press(self, event, plotID):   
    
        if event.inaxes is not None:
            self.redraw = True
            xy = (event.xdata, 0)
            
            width = 0
            height = self.canvas[plotID].axes.get_ylim()[1]
            lineColor= self.canvas[plotID].lineColor.copy()
            lineColor[3]=0.4

            self.canvas[plotID].lock = False
            self.canvas[plotID].xyInit = xy
            self.canvas[plotID].rect.set_xy(xy)
            self.canvas[plotID].rect.set_width(width)
            self.canvas[plotID].rect.set_height(height)
            self.canvas[plotID].rect.set_color(lineColor)

            self.canvas[plotID].draw()
       
        return
    
    def on_release(self, event, plotID):

        if event.inaxes is not None:

            xVals = np.sort([self.canvas[plotID].xyInit[0], event.xdata]) 
            xy = (xVals[0], 0)
            width = np.diff(xVals)
            height = self.canvas[plotID].axes.get_ylim()[1]
            lineColor= self.canvas[plotID].lineColor.copy()
            self.canvas[plotID].lock = True
            self.canvas[plotID].xRange = xVals
            
            self.canvas[plotID].rect.set_xy(xy)
            self.canvas[plotID].rect.set_width(width)
            self.canvas[plotID].rect.set_height(height)

            self.canvas[plotID].draw()
            
        else:
            self.canvas[plotID].lock = True 

        self.updateReady.emit('rangeSelected')
        return None

    def on_move(self, event, plotID):

        if (not self.canvas[plotID].lock) and (event.inaxes is not None):
            xVals = np.sort([self.canvas[plotID].xyInit[0], event.xdata]) 
            xy = (xVals[0], 0)
            width = np.diff(xVals)
            height = self.canvas[plotID].axes.get_ylim()[1]
            lineColor= self.canvas[plotID].lineColor.copy()

            self.canvas[plotID].rect.set_xy(xy)
            self.canvas[plotID].rect.set_width(width)
            self.canvas[plotID].rect.set_height(height)

            self.canvas[plotID].draw()
            # self.canvas[plotID].rect.set_color([0,0,1,0.4])

        return None 

    def resetPlot(self,plotData):


        for plotID in self.canvas.keys():

            dataDict = plotData[plotID]
            xData = dataDict['xData']
            yData = dataDict['yData']
            axesPos = dataDict['axesPos']
            lineColor = dataDict['lineColor']
            lineWidth = dataDict['lineWidth']
            
            if dataDict['drawRect']:

                self.canvas[plotID].axes.cla()
                self.canvas[plotID].axes.set_xlim(dataDict['xLim'])
                self.canvas[plotID].axes.set_ylim(dataDict['yLim'])
                
                xy = (0,0)
                width = xData[-1]
                height = self.canvas[plotID].axes.get_ylim()[-1]
                faceColor = lineColor.copy()
                edgeColor = faceColor
                faceColor[3] = 0
                self.canvas[plotID].rect.remove()
                self.canvas[plotID].rect =[]
                self.canvas[plotID].rect = mpatches.Rectangle(xy, width, height, angle = 0.0, zorder = 2, facecolor = faceColor, edgecolor = edgeColor)
                self.canvas[plotID].axes.add_patch(self.canvas[plotID].rect)
                self.canvas[plotID].xyInit = xy
                self.canvas[plotID].xRange = dataDict['xLim']
                self.canvas[plotID].lock = True

                self.canvas[plotID].axes.plot(xData, yData, color=lineColor, linewidth=lineWidth, zorder = 3)
            
            elif self.wellChanged:
                self.canvas[plotID].axes.cla()
                self.canvas[plotID].axes.set_xlim(dataDict['xLim'])
                self.canvas[plotID].axes.set_ylim(dataDict['yLim'])

                self.canvas[plotID].gcLineDir = {}
                for n in range(yData.shape[1]):
                    objID = dataDict['objID'][n]
                    self.canvas[plotID].gcLineDir[str(objID)] = self.canvas[plotID].axes.plot(xData, yData[:,n], color=lineColor, linewidth=lineWidth, zorder = 3, gid = str(objID))[0]

                self.wellChanged = False
            
            else:
                gcLineDir = self.canvas[plotID].gcLineDir

                for key, item in gcLineDir.items():
                    item.set_color(lineColor)
                    item.set_zorder(3)

            self.canvas[plotID].draw()

        return None

class Camera(object):
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.cap = None
        self.cap = cv2.VideoCapture(self.cam_num)
        self.last_frame = self.cap.read()

    def get_frame(self):
        ret, self.last_frame = self.cap.read()

        return self.last_frame

    def set_brightness(self, value):
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def get_brightness(self):
        return self.cap.get(cv2.CAP_PROP_BRIGHTNESS)

    def close_camera(self):
        self.cap.release()

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_num)

class ImageWindow(QWidget):
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
        self.trackDict = {}
        self.loadRoiData()

        self.numImages=len(self.experimentData['roiFiles'][self.roiLbl])
        self.imageNumber = 0
        self.prevImage = None

        #Create Photoviewer object
        self.viewer = OdelayStitchedImageViewer(self)

        self.gcPlots = InteractiveGCs(self.plotData)
        self.gcPlots.updateReady.connect(self.updatePlots)

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

        self.imageNumInfo = QtWidgets.QLineEdit(self)
        self.imageNumInfo.setReadOnly(True)
        
        # Add Image time slider
        self.imageSlider = QSlider(Qt.Horizontal)       
        self.imageSlider.setRange(0,self.numImages)
        self.imageSlider.sliderReleased.connect(self.changeImage)
        self.imageSlider.sliderMoved.connect(self.updateImNum)

        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout()
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
        HBlayout.addWidget(self.imageNumInfo)
        VBlayout.addLayout(HBlayout)

        HB2layout = QtWidgets.QHBoxLayout()
        HB2layout.setAlignment(QtCore.Qt.AlignLeft)
        HB2layout.addLayout(VBlayout)
        HB2layout.addWidget(self.gcPlots)

        # self.masterWidget = QWidget(self)
        # self.setCentralWidget(masterWidget)
        self.setLayout(HB2layout)

    def chooseRoi(self, ind):

        self.roiLbl = ind
        self.gcPlots.wellChanged = True
        self.numImages = len(self.experimentData['roiFiles'][self.roiLbl])
        if self.imageNumber>self.numImages:
            self.imageNumber = self.numImages
            self.imageSlider.setValue = self.numImages
    
        self.loadRoiData()
        self.loadImage()
        self.gcPlots.resetPlot(self.plotData)
        self.addTrackObj()
        self.updatePlots('plotCleared')

    def loadImage(self):
    
        self.viewer.qImage = self.readImage()
        self.addTrackObj()
        pixmap = QPixmap.fromImage(self.viewer.qImage)
        self.viewer.setPhoto(pixmap)
        self.imageNumInfo.setText('%d' % (self.imageNumber))

    def addTrackObj(self):

        imNum = self.imageNumber
        prevIm = self.prevImage
        objInds = self.plotData['dataFilter']

        objectTrack = self.roiData['objectTrack']
        objectArea = self.roiData['objectArea']

        curInds = np.arange(objectTrack.shape[0], dtype = 'int')
        updateInds  = (objectTrack[:,imNum]!=0) & (objectArea[:,imNum]!=0) & objInds
   
        curVec = curInds[updateInds]
        objVec = objectTrack[updateInds,imNum]

        if prevIm == None:
            prevInds = np.zeros((objectTrack.shape[0],), dtype = 'bool')
        else:
            totTracks = np.array([int(key) for key in self.trackDict.keys()], dtype = int)
            prevInds =  (objectTrack[:, prevIm]!=0) & (objectArea[:,prevIm]!=0)
            for n in totTracks:
                prevInds[n] = True
        
        removeInds = prevInds & ~ updateInds
        removeVec = curInds[removeInds]

        for cnt, val in enumerate(curVec, start=0):
            if str(val) in self.trackDict.keys():
                self.trackDict[str(val)].trackBox = self.imageStats[2][objVec[cnt],:]
                self.trackDict[str(val)].setPos(self.imageStats[3][objVec[cnt],0], self.imageStats[3][objVec[cnt],1])
            
            else:
                self.trackDict[str(val)] = TrackPosition(self, trackBox = self.imageStats[2][objVec[cnt],:], trackIndex = val)
                self.viewer._scene.addItem(self.trackDict[str(val)])
                self.trackDict[str(val)].trackSelected.connect(self.updatePlots)
                self.trackDict[str(val)].setPos( self.imageStats[3][objVec[cnt],0], self.imageStats[3][objVec[cnt],1])

        for val in removeVec:
            if str(val) in self.trackDict.keys():
                popVal = self.trackDict.pop(str(val))
                self.viewer._scene.removeItem(popVal)
        
        self.prevImage = imNum
        self.viewer.setScene(self.viewer._scene)

    def updatePlots(self, caller):

        plotData   = self.plotData
        fitData    = plotData['fitData']
        objID      = plotData['objID']
        inds   = np.zeros((fitData.shape[0],), dtype = 'bool')
        if isinstance(caller, int):
            selectedColor = [1,0,0,0.8]
            unselectedColor = [0.4,0.4,0.4,0.1]
            inds[objID==caller] = True

        elif caller == 'pickedInds':
            selectedColor = [1,0,0,0.8]
            unselectedColor = [0.4,0.4,0.4,0.1]
            indArray = np.array([int(val) for val in self.gcPlots.pickedList], dtype = 'int')
            indList = np.unique(indArray)
            for n in indList:
                inds = inds | (objID==n)
               
        elif caller == 'rangeSelected':
            selectedColor = [1,0,0,0.8]
            unselectedColor = [0.4,0.4,0.4,0.1]
            colKey = {'dblHist': 6, 'lagHist':5, 'texHist': 3, 'NuDHist':1 }
            inds = np.ones((fitData.shape[0],), dtype = 'bool')
            for  key, col in colKey.items():
                selectedRange = self.gcPlots.canvas[key].xRange
                col = colKey[key]
                greater = fitData[:,col]>selectedRange[0]
                lesser  = fitData[:,col]<selectedRange[1]
                inds = greater & lesser & inds

        elif caller == 'plotCleared':
            
            inds = np.ones((fitData.shape[0],), dtype = 'bool')
            selectedColor = [0,0,0,0.5]
            unselectedColor = [0.4,0.4,0.4,0.1]
            
        self.gcPlots.canvas['gcLines'].axes.set_xlim(plotData['gcLines']['xLim'])
        self.gcPlots.canvas['gcLines'].axes.set_ylim(plotData['gcLines']['yLim'])

        gcLineDir = self.gcPlots.canvas['gcLines'].gcLineDir
        trackDict = self.trackDict

        for n in objID[inds]:
            if str(n) in trackDict.keys():
                gcLineDir[str(n)].set_color(selectedColor)
                gcLineDir[str(n)].set_zorder(4)
            if str(n) in trackDict.keys():
                trackDict[str(n)].penColor = QColor(255,0,0,255)
                trackDict[str(n)].update()

        for n in objID[~inds]:
            if str(n) in gcLineDir.keys():
                gcLineDir[str(n)].set_color(unselectedColor)
                gcLineDir[str(n)].set_zorder(3)
            if str(n) in trackDict.keys():
                trackDict[str(n)].penColor = QColor(100,100,0,255)
                trackDict[str(n)].update()

        self.gcPlots.canvas['gcLines'].draw()
        self.update()

        return None

    def updateImNum(self):
        # sending_widget = self.sender()
        # self.imageNumber = sending_widget.value()
        timePoints = self.plotData['gcLines']['xData']
        timePoint = timePoints[self.imageNumber]
        self.imageNumInfo.setText('%d,  %d' % (self.imageNumber, timePoint))

    def changeImage(self):

        sending_widget = self.sender()

        if sending_widget.objectName() == self.btnNextImage.objectName():
            self.imageNumber += 1
            if self.imageNumber>self.numImages:
                self.imageNumber = self.numImages
            else:
                self.viewer.qImage = self.readImage()
                self.addTrackObj()
                pixmap = QPixmap.fromImage(self.viewer.qImage)
                self.imageSlider.setValue(self.imageNumber)
                self.viewer.setPhoto(pixmap, False)

        elif sending_widget.objectName() == self.btnPrevImage.objectName():
            self.imageNumber -= 1
            if self.imageNumber<0:
                self.imageNumber = 0
            else:
                self.viewer.qImage = self.readImage()
                self.addTrackObj()
                pixmap = QPixmap.fromImage(self.viewer.qImage)
                self.imageSlider.setValue(self.imageNumber)
                self.viewer.setPhoto(pixmap, False)

        elif sending_widget.objectName() == self.imageSlider.objectName():
            
            self.imageNumber = sending_widget.value()
            self.viewer.qImage = self.readImage()
            self.addTrackObj()
            pixmap = QPixmap.fromImage(self.viewer.qImage)
            self.viewer.setPhoto(pixmap, False)
        
        self.updateImNum()

    def pixInfo(self):
        self.viewer.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def loadRoiData(self):

        roiLbl = self.roiLbl
        imagePath = pathlib.Path(self.odelayConfig['LocalImageDir'])
        dataPath  = pathlib.Path(self.odelayConfig['LocalDataDir'])
        # Generate image file Path by combining the region of interest lable with the experiment path
        roiFolder = pathlib.Path('./'+ roiLbl)
        
        # Load Region of Interest Data.  This HDF5 file should containt location of image stitch coordinates 
        roiPath = dataPath / 'ODELAY Roi Data' / f'{roiLbl}.hdf5'

        self.roiData    = fio.loadData(roiPath)
        self.figureConfigData(self.roiData, 'Mtb')
        
    def figureConfigData(self, roiData, organism):

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
        
        pltRange = odp.setPlotRange(organism)
     
        rc = roiData['fitData'].shape
        idVec = np.arange(rc[0], dtype = 'uint32')
        inds = roiData['fitData'][:,0]>0

        roiID         = roiData['roiLabel']
        timePoints    = roiData['timePoints']/pltRange['GCs']['devisor']
        rawobjectArea = roiData['objectArea']
        rawfitData    = roiData['fitData']

        numObsv = pltRange['GCs']['numObservations']
        rngTime = pltRange['GCs']['xRange']
        rngArea = pltRange['GCs']['yRange']
        rngTdbl = pltRange['Dbl']['xRange']
        rngTlag = pltRange['Lag']['xRange']
        rngTexp = pltRange['Tex']['xRange']
        rngNDub = pltRange['NumDbl']['xRange']

        if ('roiInfo' in roiData.keys()) and (len(roiData['roiInfo'])>0):
            roiID = roiData['roiInfo']['Strain ID']

        numObservations = np.sum(rawobjectArea>0, 1) > numObsv
        numDbl   = rawfitData[:,1]>0
        fitIndex = (rawfitData[:,0]>0) & (rawfitData[:,0]<8)
        dataFilter = numObservations * fitIndex * numDbl

        fitData    = rawfitData[dataFilter, :]
        objectArea = rawobjectArea[dataFilter,:].transpose()
        objID      = idVec[dataFilter]

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

        yLim = np.array(   [rngArea,
                            [0,1],
                            [0,1],
                            [0,1],
                            [0,1]], dtype = 'float64')

        axesPos = np.array([[0.1, 0.1, 0.8, 0.8],
                            [0.125, 0.2, 0.8, 0.725],
                            [0.125, 0.2, 0.8, 0.725],
                            [0.125, 0.2, 0.8, 0.725],
                            [0.125, 0.2, 0.8, 0.725]], dtype = 'float64')
        
        lineWidth = np.array([0.8,1,1,1,1], dtype = 'float64')

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

        keys = ['gcLines', 'dblHist', 'lagHist', 'texHist','NuDHist' ]
        plotData = {}

        for cnt, key in enumerate(keys, start = 0):
            tempDict = {}
            if cnt == 0:
                tempDict['xData'] = timePoints
                tempDict['yData'] = lgobjectArea
                tempDict['objID'] = objID
                tempDict['title'] = roiID
                tempDict['drawRect'] = False

            else:
                tempDict['xData'] = nbins[cnt,:]
                tempDict['yData']  = normVirts[cnt,:]
                tempDict['drawRect'] = True

            tempDict['xLim']       = xLim[cnt,:]
            tempDict['yLim']       = yLim[cnt,:]
            tempDict['axesPos']    = axesPos[cnt,:]
            tempDict['lineColor']  = lineColor[cnt,:]
            tempDict['lineWidth']  = lineWidth[cnt]
            tempDict['key']        = key
            
            plotData[key] = tempDict
        
        plotData['fitData']     = fitData
        plotData['fitDataCols'] = roiData['fitDataCols']
        plotData['objID']       = objID
        plotData['dataFilter']  = dataFilter
        plotData['objectArea']  = objectArea

        self.plotData = plotData

    def loadExperimentData(self):

        imagePath = pathlib.Path(self.odelayConfig['LocalImageDir'])
        dataPath  = pathlib.Path(self.odelayConfig['LocalDataDir'])
        indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]

        if len(indexList)==1:
            expIndexPath = dataPath / indexList[0]
            expData = fio.loadData(expIndexPath)

        return expData

    def saveImage(self):
        location = self.odelayConfig['LocalDataDir']
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"Save Image", self.tr(location),"Images (*.png, *.jpg)", options=options)
        print(fileName)
        val = self.viewer.qImage.save(fileName, format=None, quality=100)
        if val:
            print('Image saved')

    def readImage(self, lowcut = 0.0005, highcut = 0.99995):

        roiLbl = self.roiLbl
        imNum  = self.imageNumber

        imagePath = pathlib.Path(self.odelayConfig['LocalImageDir'])
        dataPath  = pathlib.Path(self.odelayConfig['LocalDataDir'])
        # Generate image file Path by combining the region of interest lable with the experiment path
        roiFolder = pathlib.Path('./'+ roiLbl)
        imageFileName = [*self.experimentData['roiFiles'][roiLbl]]
        imageFileName.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        imageFilePath = imagePath / roiFolder / imageFileName[imNum] 

        background = self.experimentData['backgroundImage']
        # This data should be extracted from the Experiment Index file or stage data file.
        pixSize       = self.experimentData['pixSize']
        magnification = self.experimentData['magnification']

        stInd = f'{imNum:03d}'
        stitchCorners = self.roiData['stitchMeta'][stInd]['imPix']
   
        anImage = opl.assembleImage(imageFilePath, pixSize, magnification, background, stitchCorners)
        # anImage = opl.stitchImage(imageFilePath, pixSize, magnification, background)

        sobelBf   =  opl.SobelGradient(anImage['Bf'])
 
        bwBf1 = opl.morphImage(sobelBf, self.experimentData['kernalerode'], self.experimentData['kernalopen'],  self.roiData['threshold'][imNum])
        self.imageStats  = cv2.connectedComponentsWithStats(bwBf1, 8, cv2.CV_32S)
        
        # make a histogram of the image in the bitdept that the image was recorded.  
        imageHist = histogram1d(anImage['Bf'].ravel(),2**16,[0,2**16],weights = None).astype('float')

        # Calculate the cumulative probability ignoring zero values 
        cumHist = np.zeros(imageHist.shape, dtype='float')
        cumHist[1:] = np.cumsum(imageHist[1:])

        # if you expect a lot of zero set 
        cumProb = (cumHist-cumHist[0])/(cumHist[2**16-1]-cumHist[0])

        # set low and high values ot normalize image contrast.  
        loval = np.argmax(cumProb>=lowcut)
        hival = np.argmax(cumProb>=highcut)

        scIm = (anImage['Bf'].astype('float') - loval.astype('float'))/(hival.astype('float') - loval.astype('float'))*254
        lim = np.iinfo('uint8')
        scIm = np.clip(scIm, lim.min, lim.max)
        # Set image data type and make sure the array is contiguous in memory.  
        imageData = np.require(scIm, dtype = 'uint8', requirements = 'C')  
        # Set data as a QImage.  This is a greyscale image 
        Qim = QImage(imageData.data, imageData.shape[1], imageData.shape[0], imageData.shape[1], QImage.Format_Grayscale8)
                    
        Qim.data = imageData
        
        return Qim

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_C:
            self.gcPlots.resetPlot(self.plotData)
            self.addTrackObj()
            self.updatePlots('plotCleared')
        return None

class StageXYMoveGI(QGraphicsView):
    stageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(StageXYMoveGI, self).__init__(parent)
    
        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap('./images/Stage Arrows.jpg')

        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setMinimumSize(self._pixmap.width(), self._pixmap.height())
        self.setSceneRect(0, 0, self._pixmap.width(), self._pixmap.height())

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)

    def fitInView(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                        viewrect.height() / scenerect.height())
            self.scale(factor, factor)
        
    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.stageClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(StageXYMoveGI, self).mousePressEvent(event)

class StageZMoveGI(QGraphicsView):
    focusClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(StageZMoveGI, self).__init__(parent)
    
        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap('./images/Focus Control.jpg')

        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setMinimumSize(self._pixmap.width(), self._pixmap.height())
        self.setSceneRect(0, 0, self._pixmap.width(), self._pixmap.height())

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)

    def fitInView(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                        viewrect.height() / scenerect.height())
            self.scale(factor, factor)
        
    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.focusClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(StageZMoveGI, self).mousePressEvent(event)

class StageNavigation(QGraphicsView):
    stageNavigationSignal = pyqtSignal(QPoint)

    def __init__(self, parent, mP, dictRef):
        super(StageNavigation, self).__init__(parent)
    
        self.dictRef = dictRef
        self.setMinimumSize(300,200)
        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap(QSize(mP[dictRef]['sceneDim'][0]/10, mP[dictRef]['sceneDim'][1]/10))
        self._pixmap.fill(QColor('blue'))
        
        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
    
        # Improve this section for experiment Properties
        wellDiameter = round(mP[dictRef]['wellDiameter']/10)
        xOffset = round(mP[dictRef]['wellSpacing']/10)
        yOffset = round(mP[dictRef]['wellSpacing']/10)
        
        roiDict  = mP[dictRef]['roiDict']
        roiList  = [*roiDict]
        roiOrder = mP[dictRef]['roiOrder']
  
        
        self.posMarker = []
        cntr = 0
        for roi in roiList:
            self.posMarker.append(PlatePosition(self, wellDiameter))
            xyPos = roiDict[roi]['xyPos']
            xPos = round(xyPos[0]/10)+xOffset
            yPos = round(xyPos[1]/10)+yOffset
            self.posMarker[-1].setPos(xPos, yPos)
            self.posMarker[-1].posIndex = cntr
            self.posMarker[-1].call_Position.connect(self.moveStage)
            self._scene.addItem(self.posMarker[-1])
            cntr+=1
    
        for ind in roiOrder:
            self.posMarker[ind].color = QColor(255,0,0,255)
            self.posMarker[ind].activePos = True

        self.imageMarker = ImagePosition(self)
        self.imageMarker.setPos(xOffset,yOffset)
        self._scene.addItem(self.imageMarker)

        self.setScene(self._scene)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(255,255, 255,255)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)
        # self.setDragMode(QGraphicsView.ScrollHandDrag)
        
    def moveStage(self, index):

        markerIndex = [x for x in range(len(self.posMarker))]
        markerIndex.pop(index)
        
        for ind in markerIndex:
            if self.posMarker[ind].activePos:
                self.posMarker[ind].color = QColor(255, 0, 0, 255)
            else:
                self.posMarker[ind].color = QColor(100, 100, 100, 255)
            self.posMarker[ind].update()

        x = self.posMarker[index].x()
        y = self.posMarker[index].y()
   
        # print(f'index of spot is {index}')
        # print(f'position of spot is {x}, {y}')
        
    def mousePressEvent(self, event):
      
        if self._photo.isUnderMouse():
            self.stageNavigationSignal.emit(self.mapToScene(event.pos()).toPoint())
        super(StageNavigation, self).mousePressEvent(event)

class ImagePosition(QGraphicsObject):
    give_Position = pyqtSignal(int)
    def __init__(self, parent):
        super(ImagePosition, self).__init__()
        self.color = QColor(0,0,150,100)
        self.imageSize = QRectF(-67,-67, 134, 134)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def boundingRect(self):
        return QRectF(self.imageSize)

    def paint(self, painter, option, widget):
        painter.setPen(QPen(Qt.blue, 5))
        painter.setBrush(QBrush(self.color))
        painter.drawRect(self.imageSize)

class PlatePosition(QGraphicsObject):
    call_Position = pyqtSignal(int)
    def __init__(self, parent, wellDiameter=450):
        super(PlatePosition, self).__init__()
        self.color = QColor(100,100,100,255)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.wellDiameter = wellDiameter
        self.activePos = False
        self.posIndex = 0
        
    def boundingRect(self):
        return QRectF(-self.wellDiameter/2, -self.wellDiameter/2, self.wellDiameter, self.wellDiameter)
    
    def paint(self, painter, option, widget):
        painter.setPen(QPen(Qt.black, 20))
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(-self.wellDiameter/2, -self.wellDiameter/2, self.wellDiameter, self.wellDiameter)
        
    def mousePressEvent(self,event):
        
        self.color = QColor(0,255,0,255)
        self.call_Position.emit(self.posIndex)
        super(PlatePosition, self).mousePressEvent(event)

        self.update()

class MicroscopeControl(QWidget):
   
    def __init__(self, mP):
        super(MicroscopeControl, self).__init__()

        # functions to create graphic groups.  Each function creates an element of the main control pannel
        # self.createMenus()
        
        self.createStageControl(mP)
        self.createAutofocus(mP)
        self.createCameraControl(mP)
        self.createImageDisplay(mP)
        
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.pickerImageDisplay, 0, 0, 2, 2)
        mainLayout.addWidget(self.stageControl,       0, 2, 2, 2)
        mainLayout.addWidget(self.focusControl,       0, 4, 1, 1)
        mainLayout.addWidget(self.cameraControl,      1, 4, 1, 1)

        mainLayout.setColumnStretch(0,5)
        mainLayout.setColumnStretch(1,5)
        mainLayout.setColumnStretch(2,5)
        mainLayout.setColumnStretch(3,5)
        mainLayout.setColumnStretch(3,5)
        mainLayout.setColumnStretch(4,1)
        
        self.setLayout(mainLayout)

        self.setWindowTitle("Microscope Control")

    def createStageControl(self, mP):

        self.stageControl  = QGroupBox('Stage Control')
        self.stageLayout   = QGridLayout()
        self.stageMove     = StageXYMoveGI(self)
        self.stageMove.stageClicked.connect(self.stageClicked)
        self.stageFocus    = StageZMoveGI(self)
        self.stageFocus.focusClicked.connect(self.focusClicked)

        self.stageNavigate = StageNavigation(self, mP, 'Source Dict')
        self.stageNavigate.stageNavigationSignal.connect(self.navClicked)
        
        self.receiverNavigate = StageNavigation(self, mP, 'Receiver Dict')
       
        self.navPixInfo = QLineEdit(self)
        self.navPixInfo.setMinimumWidth(self.stageMove._pixmap.width())
        self.navPixInfo.setReadOnly(True)

        self.xOrigin = QLineEdit(self)
        # self.xOrigin.setMinimumWidth(50)
        self.yOrigin = QLineEdit(self)
        # self.yOrigin.setMinimumWidth(50)
        self.zOrigin = QLineEdit(self)
        # self.zOrigin.setMinimumWidth(50)

        self.xStagePos = QLineEdit(self)
        # self.xStagePos.setMinimumWidth(50)
        self.yStagePos = QLineEdit(self)
        # self.yStagePos.setMinimumWidth(50)
        self.zStagePos = QLineEdit(self)
        # self.zStagePos.setMinimumWidth(50)

        self.setOrigin   = QPushButton(self)
        self.setOrigin.setText('Set Org')
        # self.setOrigin.setMinimumWidth(50)

        self.measureDist   = QPushButton(self)
        self.measureDist.setText('Measure')
        # self.measureDist.setMinimumWidth(50)
        self.measureDist.setCheckable(True)

        self.startODELAY = QPushButton(self)
        self.startODELAY.setText('ODELAY!')
        # self.startODELAY.setMinimumWidth(100)
        self.startODELAY.setCheckable(True)

        # row, col , rowspan, colspan
        self.stageLayout.addWidget(self.stageMove,      0,0,1,1)
        self.stageLayout.addWidget(self.stageFocus,     0,1,1,1)
        self.stageLayout.addWidget(self.startODELAY,    5,0,1,1)

        self.stageLayout.addWidget(self.navPixInfo,     4,0,1,1)
        self.stageLayout.addWidget(self.xOrigin,        1,0,1,1)
        self.stageLayout.addWidget(self.yOrigin,        2,0,1,1)
        self.stageLayout.addWidget(self.zOrigin,        3,0,1,1)

        self.stageLayout.addWidget(self.setOrigin,      4,1,1,1)
        self.stageLayout.addWidget(self.measureDist,    5,1,1,1)
        
        self.stageLayout.addWidget(self.xStagePos,      1,1,1,1)
        self.stageLayout.addWidget(self.yStagePos,      2,1,1,1)
        self.stageLayout.addWidget(self.zStagePos,      3,1,1,1)
        
        self.stageLayout.addWidget(self.stageNavigate,     0,3,3,6)
        self.stageLayout.addWidget(self.receiverNavigate,  4,3,3,6)
        self.stageControl.setLayout(self.stageLayout)

    def createIllumination(self, mP):

        self.illumControl  = QGroupBox('Illumination')
        illumLayout   = QGridLayout()
       
        # Attempting to set up a dynamic layout where if the 
        # device controled by the widget is present the widget 
        # is created and drawn.  It doesn't look good in the 
        # coding layout but works.

        # Check for Transmitted Light controller
        self.transIllum   = QSlider(Qt.Vertical)
        self.transEdit     = QLineEdit()
        illumLayout.addWidget(self.transIllum,    0,0,5,1, Qt.AlignCenter)
        illumLayout.addWidget(self.transEdit,     6,0,1,1, Qt.AlignCenter)

        self.condenserApp = QSlider(Qt.Vertical)
        self.condenserEdit = QLineEdit()
        illumLayout.addWidget(self.condenserApp,  0,1,5,1, Qt.AlignCenter)
        illumLayout.addWidget(self.condenserEdit, 6,1,1,1, Qt.AlignCenter)
        
        self.fieldDia     = QSlider(Qt.Vertical)
        self.fieldEdit     = QLineEdit()
        illumLayout.addWidget(self.fieldDia,      0,2,5,1, Qt.AlignCenter)
        illumLayout.addWidget(self.fieldEdit,     6,2,1,1, Qt.AlignCenter)
        
        self.transShutter = QPushButton()
        self.transShutter.setCheckable(True)
        self.transShutter.setText('Trans')
        self.transShutter.setMinimumSize(QSize(80,100))
        illumLayout.addWidget(self.transShutter,  0,3,3,2, Qt.AlignCenter)

        self.fluorShutter = QPushButton()
        self.fluorShutter.setCheckable(True)
        self.fluorShutter.setText('Fluor')
        self.fluorShutter.setMinimumSize(QSize(80,100))
        illumLayout.addWidget(self.fluorShutter,  4,3,3,2, Qt.AlignCenter)

        self.illumControl.setLayout(illumLayout)

    def createAutofocus(self, mP):
        self.focusControl = QGroupBox('Auto Focus')
        autofocusLayout   = QGridLayout()

        self.autoFocus_0  = QPushButton()
        self.autoFocus_0.setObjectName('autoFocus_0')
        self.autoFocus_0.setText('Auto Focus 1')
        
        self.focusRange_0 = QLineEdit()
        self.focusRange_0.setText(str(mP['AutoFocus_0']['zRange']))
        
        self.numSteps_0   = QLineEdit()
        self.numSteps_0.setText(str(mP['AutoFocus_0']['numSteps']))
        
        self.targetInc_0  = QLineEdit()
        self.targetInc_0.setText(str(mP['AutoFocus_0']['targetInc']))
      
        autofocusLayout.addWidget(self.autoFocus_0,  0, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.focusRange_0, 1, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.numSteps_0,   2, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.targetInc_0,  3, 0, 1, 1, Qt.AlignCenter)

        self.focusControl.setLayout(autofocusLayout)

    def createCameraControl(self, mP):
        self.cameraControl  = QGroupBox('Camera')
        cameraLayout   = QVBoxLayout()

        self.recordButton = QPushButton()
        self.recordButton.setCheckable(True)
        self.recordButton.setText('Record')
        self.focusButton  = QPushButton()
        self.focusButton.setCheckable(True)
        self.focusButton.setText('Focus')
        self.setDirButton = QPushButton()
        self.setDirButton.setCheckable(True)
        self.setDirButton.setText('Set Dir')
        self.imageNameEdit = QLineEdit()

        # self.recordButton.clicked.connect(self.cameraRecord)
        # self.focusButton.clicked.connect(self.cameraFocus)
        # self.setDirButton.clicked.connect(self.setImageDirectory)
        
        cameraLayout.addWidget(self.recordButton)
        cameraLayout.addWidget(self.focusButton)
        cameraLayout.addWidget(self.setDirButton)
        cameraLayout.addWidget(self.imageNameEdit)
        self.cameraControl.setLayout(cameraLayout)

        pass

    def createImageDisplay(self, mP):

        self.pickerImageDisplay = QGroupBox('Picker Image')
        self.displayLayout = QHBoxLayout()
        self.usbImageDisplay = MicroscopeImageDisplay(self)
        self.histDisplay = HistogramDisplay(self)

        self.displayLayout.addWidget(self.usbImageDisplay)
        self.displayLayout.addWidget(self.histDisplay)
        self.displayLayout.setStretch(0,9)
        self.displayLayout.setStretch(0,1)

        self.pickerImageDisplay.setLayout(self.displayLayout)

    def imageUpdate(self, cameraImage):

        self.usbImageDisplay.imageUpdate(cameraImage)
        self.histDisplay.generateHist(cameraImage)

    def setImageDirectory(self):
        pass

    def setImageName(self):
        pass
        
    def selectRecord(self):
        
        return None

    def stageClicked(self, pos):
        # add routine for moving stage here
        self.navPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def focusClicked(self,pos):

        self.navPixInfo.setText('%d, %d' % (pos.x(), pos.y()))
        
        return None

    def navClicked(self, pos):
        # add routine for moving stage here
        self.navPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def callPosition(self, index):
        print(f'The spot clicked is {index}')

    def navUpdate(self):
        # update the navigation plane to draw new images or show stage position
        self.stageNavigate.fitInView(self.stageNavigate.sceneRect(),1)
        self.receiverNavigate.fitInView(self.receiverNavigate.sceneRect(),1)

class DisplayPanel(QWidget):
    
    def __init__(self):
        super(DisplayPanel, self).__init__()
          
        # TODO: bring in mP to set initial sensor size
        self.imageData = np.zeros((2048,2048), dtype = 'uint16')
        
        displayLayout = QGridLayout()

        self.imageDisplay = MicroscopeImageDisplay(self)
        self.histDisplay = HistogramDisplay(self)

        self.displayLayout.addWidget(self.imageDisplay, 0,0)
        mainLayout.addWidget(self.histDisplay,  0,1)

        self.Layout.setLayout(mainLayout)

    def imageUpdate(self, cameraImage):

        self.imageDisplay.imageUpdate(cameraImage)
        self.histDisplay.generateHist(cameraImage)
       
class HistogramDisplay(QGraphicsView):
    
    def __init__(self, parent):
        super(HistogramDisplay, self).__init__(parent)
        height = self.height()
        width  = self.width()
        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap(QSize(width, height))
        self._pixmap.fill(QColor(0,0,0,255))

        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.sceneCounter = 0

        self.setSceneRect(0, 0, self._pixmap.width(), self._pixmap.height())

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)
        # self.fitInView()
       
    def fitInView(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                         viewrect.height() / scenerect.height())
            self.scale(factor, factor)

    def setPlot(self, reset=True):
        height = self.height()
        width  = self.width()
        self._pixmap = QPixmap(QSize(width,height))
        self._pixmap.fill(QColor(50,50,50,255))
        painter = QPainter()
        painter.begin(self._pixmap)
        
        # set up gradient brush
        gradient = QLinearGradient(QPointF(0, 0), QPointF(0, self._pixmap.width()))
        gradient.setColorAt(0, QColor(255,0,0))
        
        brush = QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)

        # Draw the paths for the chart
        path = QPainterPath()
        path.moveTo(1,1)
        for indx, value in enumerate(self.normHist, 1):
            path.lineTo(value, indx)
    
        painter.drawPath(path)
        painter.end()
        self._photo.setPixmap(self._pixmap)

        self.fitInView()
       
    def generateHist(self, cameraImage):
        nbins = self._pixmap.height()
        data_range  = self._pixmap.width()
        binnedImage = cv2.resize(cameraImage, (512, 512))
        imHist = histogram1d(binnedImage.ravel(),nbins,[0,2**16],weights = None).astype('float')
        imHist[0]  = 0
        imHist[-1] = 0
        value_fraction = 0.95 * imHist / (np.max(imHist)+1)
        self.normHist = np.round(value_fraction * data_range)
        self.setPlot()

class MicroscopeImageDisplay(QGraphicsView):
    # TODO:  Add in zoom and plotting for reviewing data.
    def __init__(self, parent):
        super(MicroscopeImageDisplay, self).__init__(parent)

        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._qImg   = QImage()
        self._pixmap = QPixmap(QSize(2048, 2048))
        self._pixmap.fill(QColor(50,50,50,255))
        
        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        # self.setFixedSize(500,500)
        self.setSceneRect(0, 0, self._pixmap.width(), self._pixmap.height())

        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)

    def imageUpdate(self, cameraImage,  lowcut = 0.0005, highcut = 0.99995):

        imageHist = histogram1d(cameraImage.ravel(),2**16,[0,2**16],weights = None).astype('float')
       
        # Calculate the cumulative probability ignoring zero values 
        cumHist = np.zeros(imageHist.shape, dtype='float')
        cumHist[1:] = np.cumsum(imageHist[1:])

        cumRange = cumHist[2**16-1]-cumHist[0]
        # if you expect a lot of zero set 
        cumHist-=cumHist[0]
        cumHist /=cumRange

        # set low and high values ot normalize image contrast.  
        loval = np.argmax(cumHist>=lowcut)
        hival = np.argmax(cumHist>=highcut)
   
        im = np.clip(cameraImage, loval, hival).astype('float')
        scaleFactor = 254/(hival-loval)
        im -=loval
        im *= scaleFactor
        # Set image data type and make sure the array is contiguous in memory.  
        imageData = np.require(im, dtype = 'uint8', requirements = 'C')
        
        self._qImg = QImage(imageData.data, imageData.shape[1], imageData.shape[0], imageData.shape[1], QImage.Format_Grayscale8)
        self._pixmap = QPixmap(self._qImg)
        self.setPhoto(self._pixmap)
       
    def fitInView(self):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                        viewrect.height() / scenerect.height())
            self.scale(factor, factor)
           
    def setPhoto(self, pixmap=None, reset=True):

        if pixmap and not pixmap.isNull():
            self._photo.setPixmap(pixmap)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        if reset:
            self.fitInView()
    
class FocusPlot(QWidget):
    def __init__(self, FocusData):
        super(FocusPlot, self).__init__()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.axis = self.figure.add_subplot(111)

        self.layoutVertical = QtWidgets.QVBoxLayout(self)#QVBoxLayout
        self.layoutVertical.addWidget(self.canvas)
        self.setLayout(self.layoutVertical)

        xData = FocusData['xData']
        yData = FocusData['yData']

        self.axis.plot(xData, yData, color = [0, 0, 1], linewidth = 1)
        self.axis.set_xlabel('Z-Position', fontweight='bold')
        self.axis.set_ylabel('Focus Score', fontweight='bold')

        self.canvas.draw()

class MarlinUSBInterface(object):
    def __init__(self):
        super(object, self).__init__()
        port = None
        buad = 9600
        self.serialPort = serial.Serial('/dev/ttyUSB0', 115200, exclusive = True)
        lines = self.serialPort.read(self.serialPort.in_waiting).decode().splitlines(False)
        line = line + "\n"
        self.serialPort.write(line.encode())


        # self.homedir   =  pathlib.Path.home()
        # self.configfile = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

    # Open serial port

    # s = serial.Serial(args.port,115200)
    # print('Opening Serial Port')
    
    # # Open g-code file
    # #f = open('/media/UNTITLED/shoulder.g','r');
    # f = open(args.file,'r');
    # print 'Opening gcode file'
    
    # # Wake up 
    # s.write("\r\n\r\n") # Hit enter a few times to wake the Printrbot
    # time.sleep(2)   # Wait for Printrbot to initialize
    # s.flushInput()  # Flush startup text in serial input
    # print 'Sending gcode'
    
    # # Stream g-code
    # for line in f:
    #     l = removeComment(line)
    #     l = l.strip() # Strip all EOL characters for streaming
    #     if  (l.isspace()==False and len(l)>0) :
    #         print 'Sending: ' + l
    #         s.write(l + '\n') # Send g-code block
    #         grbl_out = s.readline() # Wait for response with carriage return
    #         print ' : ' + grbl_out.strip()
  
    # # Close file and serial port
    # f.close()
    # s.close()

class MasterControl(QMainWindow):
    '''
    MasterControl is the base window for our controller
    '''

    def __init__(self):
        super(MasterControl, self).__init__()

        # TODO: Create window with 
        # Load Dataset
        # Connect printer
        # Load parsing file or create parsing file
        # create movement file
        # Movement loop
            # calibrate
            # move to location
            # fine toon location
            # pick colony
            # deposit colony
            # log file
            # repeat 
        
        # self.createMenus()
        
        
        self.primarySaveDir = None
        self.backupSaveDir  = None
        self.microscopeConfigFilePath = None
        self.autoFocusParams = {}

        self.setWindowTitle('ODELAY Picking!!!!')

        self.text_DataDir = QLabel()
        self.text_DataDir.setText('Data Directory')

        self.ledit_DataDirectory   = QLineEdit()
        self.button_ChooseDataDir  = QPushButton()
        self.button_ChooseDataDir.setText('Data Dir')
        self.button_ChooseDataDir.clicked.connect(self.chooseDataDir)

        self.text_ImageDir = QLabel()
        self.text_ImageDir.setText('Image Directory')
        self.ledit_ImageDirectory  = QLineEdit()

        self.button_ChooseImageDir = QPushButton()
        self.button_ChooseImageDir.setText('Image Dir')
        self.button_ChooseImageDir.clicked.connect(self.chooseImageDir)

        self.text_ExperimentType = QLabel()
        self.text_ExperimentType.setText('Experimet Type')
        self.selectExperimentType   = QComboBox()
        experimentList = ['ODELAY 96 Spot', 'ODELAY 5 Condition', 'External File']
        self.selectExperimentType.addItems(experimentList)
        self.selectExperimentType.currentTextChanged.connect(self.selectExperiment)
        self.button_ExperimentType = QPushButton()
        self.button_ExperimentType.setText('Exp File')

        self.text_MicroscopeConfig = QLabel()
        self.text_MicroscopeConfig.setText('MM Config File')
        self.ledit_MicroscopeConfig = QLineEdit()
        self.button_MicroscopeConfig = QPushButton()
        self.button_MicroscopeConfig.setText('Config File')
        self.button_MicroscopeConfig.clicked.connect(self.chooseMicroscopeConfigFile)


        self.launchButton = QPushButton()
        self.launchButton.setText('Launch')
        qss= f'QPushButton{{background-color:rgb(255,0,0); font-size: 10pt; font-weight: bold}}'
        self.launchButton.setStyleSheet(qss)

        masterGridLayout = QGridLayout()
        masterGridLayout.setColumnStretch(0,1)
        masterGridLayout.setColumnStretch(1,3)
        masterGridLayout.setColumnStretch(2,1)
    
        masterGridLayout.addWidget(self.text_DataDir,         1,0,1,1)
        masterGridLayout.addWidget(self.ledit_DataDirectory,  1,1,1,1)
        masterGridLayout.addWidget(self.button_ChooseDataDir, 1,2,1,1) 

        masterGridLayout.addWidget(self.text_ImageDir,          2,0,1,1)
        masterGridLayout.addWidget(self.ledit_ImageDirectory,   2,1,1,1)
        masterGridLayout.addWidget(self.button_ChooseImageDir,  2,2,1,1) 

        masterGridLayout.addWidget(self.text_MicroscopeConfig,   3,0,1,1)
        masterGridLayout.addWidget(self.ledit_MicroscopeConfig,  3,1,1,1)
        masterGridLayout.addWidget(self.button_MicroscopeConfig, 3,2,1,1)

        masterGridLayout.addWidget(self.text_ExperimentType,     4,0,1,1)
        masterGridLayout.addWidget(self.selectExperimentType,    4,1,1,1)
        masterGridLayout.addWidget(self.button_ExperimentType,   4,2,1,1)

        masterGridLayout.addWidget(self.launchButton,            5,1,1,1)
        
        self.launchButton.clicked.connect(self.createInterface)

        masterWidget=QWidget()

        masterWidget.setLayout(masterGridLayout)
        self.setCentralWidget(masterWidget)
        self.setGeometry(500, 300, 1000, 300)
        self.show()

        self.imageDir = pathlib.Path('D:\\Python_Projects\\Example Data\\H37Rv OFX VER')
        self.dataDir  = pathlib.Path('D:\\Python_Projects\\Example Data\\OFX v3 Processed 2020-02-01')

        self.createInterface()

    def selectExperiment(self, modName = None):
        if modName == None:
            modName = self.selectExperimentType.currentText()

        mP = self.microscopeDefaultConfig() 
        expDict = expMods.moduleSelector(modName, mP)

        self.mP = expDict
    

        return None

    def chooseDataDir(self):

        options = QFileDialog.Options()
        dirName = QFileDialog.getExistingDirectory(self,"Select ODELAY Primary Directory", "", options=options)

        if len(dirName)>0:
            primarySavePath = pathlib.Path(dirName)
            self.ledit_PrimaryDirectory.setText(str(primarySavePath))
            self.primarySaveDir = primarySavePath
            if self.backupSaveDir == None:
                qss= f'QPushButton{{background-color:rgb(255,255,0); font-size: 10pt; font-weight: bold}}'

        return None

    def chooseImageDir(self):

        options = QFileDialog.Options()
        dirName = QFileDialog.getExistingDirectory(self,"Select ODELAY Backup Directory", "", options=options)

        if len(dirName)>0:
            imagePath = pathlib.Path(dirName)
            self.ledit_ImageDirectory.setText(str(imagePath))
            self.imageDir = imagePath

            if (self.primarySaveDir != None) and self.primarySaveDir.exists() and self.backupSaveDir.exists():
                qss= f'QPushButton{{background-color:rgb(255,255,0); font-size: 10pt; font-weight: bold}}'
                self.launchButton.setStyleSheet(qss)

        return None

    def chooseMicroscopeConfigFile(self):

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Microscop Config File", "", options=options)
        print(fileName)
        
        if len(fileName)>0:
            self.microscopeConfigFilePath = pathlib.Path(fileName)
            self.ledit_MicroscopeConfig.setText(fileName)

            if ((self.primarySaveDir != None) and self.primarySaveDir.exists() and 
                    (self.backupSaveDir != None) and self.backupSaveDir.exists() and 
                    (self.microscopeConfigFilePath != None) and self.microscopeConfigFilePath.exists()):
                qss= f'QPushButton{{background-color:rgb(0,255,0); font-size: 10pt; font-weight: bold}}'
                self.launchButton.setStyleSheet(qss)

        return None

    def createInterface(self):
            '''
            Generates 2 GUI panels, the camera display panel and the microscope control panel.
            ''' 
            self.mP = self.microscopeDefaultConfig()

            self.connectPickerCamera()
  
            self.loadMicroscopeConfig()
            
            self.updateTimer = QTimer()
            self.updateTimer.setInterval(250)

            self.videoTimer  = QTimer()
            self.videoTimer.setInterval(50)
            self.videoTimer.timeout.connect(self.snapImage)

            self.writeTimer = QTimer()
            self.writeTimer.setInterval(5000)
            
            self.odelayTimer = QTimer()

            self.controlPanel = MicroscopeControl(self.mP)
            self.controlPanel.setGeometry(200, 200, 1500, 600)
            self.controlPanel.show()
            self.controlPanel.navUpdate()

            self.displayPanel = ImageWindow()
            self.displayPanel.setGeometry(100, 100, 800, 500)
            self.displayPanel.show()
            self.displayPanel.loadImage()

            # # Connect Signals to Slots here for ordered control of instrument.

            # self.controlPanel.focusButton.clicked.connect(self.videoControl)

            # self.controlPanel.autoFocus_0.clicked.connect(self.autoFocusButton)
            # self.controlPanel.autoFocus_1.clicked.connect(self.autoFocusButton)

            # self.controlPanel.stageMove.stageClicked.connect(self.guiMoveXYStage)
            # self.controlPanel.stageFocus.focusClicked.connect(self.guiMoveZStage)

            # for roiInd in range(len(self.controlPanel.stageNavigate.posMarker)):
            #     self.controlPanel.stageNavigate.posMarker[roiInd].call_Position.connect(self.navMoveStage)


            # self.snapImage()
    
    def openConfigFileDialog(self):
    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Microscope Config File", "","Micro-M  (*.cfg);;", options=options)
        return fileName

    def saveDirectoryDialog(self):
    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getExistingDirectory(self,"Select Image Directory", "","", options=options)
        return fileName

    def connectPickerCamera(self):

    
        # self.pickerCamera = Camera(0)


        return None

    def connectPrinter(self):
        ''' Add USB handshaking protocol'''
        return None
        
    def generateROIDict(self, wellSpacing=4500, wellDiameter=4500, rowNames = ['E','F','G','H','I','J','K','L'], colNames =['06','07','08','09','10','11','12','13','14','15','16','17','18','19'], colImaged =  np.arange(1,len(['06','07','08','09','10','11','12','13','14','15','16','17','18','19'])-1)):

        expDict={}
        expDict['wellSpacing']  =  wellSpacing
        expDict['wellDiameter'] = wellDiameter
        expDict['rowName'] = rowNames
        expDict['colName'] = colNames
        columnList = colImaged

        wellSpacing = expDict['wellSpacing']
        numRows = len(expDict['rowName'])
        numCols = len(expDict['colName'])
        expDict['sceneDim']=[wellSpacing*(numCols+1), wellSpacing*(numRows+1)]

        expDict['xgrid']   = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()
        expDict['ygrid']   = np.arange(0,wellSpacing*numCols,wellSpacing,dtype='float').tolist()

        cnt = 0
        order = 1
       
        roiOrder  = []    
        stageXYPos = {}
        roiXYPos = {}
        roiDict    = {}
        for row, rowID in enumerate(expDict['rowName'],0):
            for col, colID in enumerate(expDict['colName'],0):
          
                roiXYPos[f'{rowID}{colID}']       = [expDict['xgrid'][col], expDict['ygrid'][row], cnt]
                roiDict[f'{rowID}{colID}'] = {}
                roiDict[f'{rowID}{colID}']['xyPos'] = [expDict['xgrid'][col], expDict['ygrid'][row]]
                cnt+=1
            columnVec = columnList[::order]+row*len(expDict['colName'])
            roiOrder.extend(columnVec.tolist())
            order *= -1

        expDict['roiOrder']   = roiOrder
        expDict['stageXYPos'] = stageXYPos
        expDict['roiDict']    = roiDict
        expDict['roiList']    = [[*roiDict][roiInd] for roiInd in roiOrder]
        expDict['roiXYPos']   = roiXYPos

        return expDict

    def microscopeDefaultConfig(self):
        mP = {}

        #  AutoFocus Props
        mP['expInitialized'] = False

        mP['AutoFocus_0'] = {}
        mP['AutoFocus_0']['zRange']    = 80
        mP['AutoFocus_0']['numSteps']  = 11
        mP['AutoFocus_0']['targetInc'] = 1

        mP['AutoFocus_1'] = {}
        mP['AutoFocus_1']['zRange']    = 20
        mP['AutoFocus_1']['numSteps']  = 7
        mP['AutoFocus_1']['targetInc'] = 0.3

        mP['Source Dict']   = self.generateROIDict(wellSpacing=4500, wellDiameter=4500, rowNames = ['E','F','G','H','I','J','K','L'], colNames =['06','07','08','09','10','11','12','13','14','15','16','17','18','19'], colImaged =  np.arange(1,len(['06','07','08','09','10','11','12','13','14','15','16','17','18','19'])-1))
        mP['Receiver Dict'] = self.generateROIDict(wellSpacing=9000, wellDiameter=9000, rowNames = ['A','B','C','D','E','F','G','H'], colNames =['01','02','03','04','05','06','07','08','09','10','11','12'], colImaged =  np.arange(0,len(['01','02','03','04','05','06','07','08','09','10','11','12'])))
      
        
        #  Microscope Conditions              
        mP['XYZOrigin'] = [35087,26180 , 6919];      
     
        # Load dictionary that will define microscope states based on configuration loaded.
        # Brightfield Image Conditions

        return mP

    def loadMicroscopeConfig(self, filePath=None):
        '''
        TODO:
        1.  load choosen config or default configDictionary
        2.  Determin microscope file 
        3.  return file for user interface
        '''
      
        return None

    def loadExcelMicroscopeConfig(self, filePath):

        configDict = fio.readMMConfigFile(filePath)

    def videoControl(self):
        
        return None

    def snapImage(self):

        return None

    def recordImage(self):
      
        return None

    def navMoveStage(self, posIndex):
    
        return None
     
    def guiMoveXYStage(self,movePos):

        return None

    def guiMoveZStage(self,Pos):
        
        return None

    def moveXYStage(self, xPos, yPos):
        '''
        Handles moving the stage XY positions.   
        '''

        return None
    
    def moveZStage(self, zPos):
        
    
        return None

    def selectRecord(self):

        pass

    def updateControlPanel(self):

        self.absXYZPos = np.array(([self.mmc.getXPosition(self.mP['xyDrive']), 
                                    self.mmc.getYPosition(self.mP['xyDrive']),
                                    self.mmc.getPosition(self.mP['zDrive'])]),dtype = 'float') 
        
        self.relXYZPos = self.absXYZPos*self.xyzDir + np.array(self.mP['XYZOrigin'], dtype = 'float')

        xyz  = np.round(self.relXYZPos,decimals = 2)  
        xyzO = self.mP['XYZOrigin']
        
        self.controlPanel.xOrigin.setText(str(xyzO[0])) 
        self.controlPanel.yOrigin.setText(str(xyzO[1])) 
        self.controlPanel.zOrigin.setText(str(xyzO[2]))

        self.controlPanel.xStagePos.setText(str(xyz[0])) 
        self.controlPanel.yStagePos.setText(str(xyz[1]))
        self.controlPanel.zStagePos.setText(str(xyz[2]))

        xOffset = round(self.mP['Source Dict']['wellSpacing']/10)
        yOffset = xOffset
        
        xPos = round(xyz[0]/10) + xOffset
        yPos = round(xyz[1]/10) + yOffset

        markerSize = np.round(np.array(self.cameraImage.shape)*self.mmc.getPixelSizeUm()/10)

        self.controlPanel.stageNavigate.imageMarker.imageSize = QRectF(-round(markerSize[1]/2),-round(markerSize[0]/2),markerSize[1], markerSize[0])
        self.controlPanel.stageNavigate.imageMarker.setPos(xPos, yPos)

        return None

    def autoFocusButton(self):
        sendingButton = self.sender()
        if sendingButton.objectName() == 'autoFocus_0':
            self.setautoFocusParams(0)

        elif sendingButton.objectName() == 'autoFocus_1':
            self.setautoFocusParams(1)
            
        self.autoFocus()

    def setautoFocusParams(self, focusPhase):
        
        if   focusPhase == 0:
            self.autoFocusParams['zRange']   = int(self.controlPanel.focusRange_0.text())
            self.autoFocusParams['numSteps'] = int(self.controlPanel.numSteps_0.text())
            self.autoFocusParams['targetInc']= float(self.controlPanel.targetInc_0.text())

        elif focusPhase == 1:
            self.autoFocusParams['zRange']   = int(self.controlPanel.focusRange_1.text())
            self.autoFocusParams['numSteps'] = int(self.controlPanel.numSteps_1.text())
            self.autoFocusParams['targetInc']= float(self.controlPanel.targetInc_1.text())        

        return None

    def autoFocus(self):
        '''Auto Focus Routine.  Requires autofocus paparmeters to be set before calling to ensure they are updated to the correct condition'''

        videoOn      = self.controlPanel.focusButton.isChecked()
        videoRunning = self.videoTimer.isActive()
        if videoRunning:
            self.videoTimer.stop()
        
        # Figure out who coalled and check focus ranges
        zRange    = self.autoFocusParams['zRange']
        numSteps  = self.autoFocusParams['numSteps']
        targetInc = self.autoFocusParams['targetInc']
        
        # Calculate Z-Movements
        zPos = self.mmc.getPosition(self.mP['zDrive'])
        zRecord = np.zeros((2*numSteps+21,3), dtype = 'float')
        upZ  = zPos+zRange
        lowZ = zPos-zRange
    
        zFocus = np.zeros((numSteps+3,3), dtype = 'float')    
        zFocus[:numSteps,0] = np.linspace(lowZ,upZ,numSteps, dtype = 'float')
        zIncrement = np.absolute(zFocus[1,0]-zFocus[0,0])
        recInd = 1
        zPhase = 1
        
        for step in range(numSteps):
            zFocusPos = zFocus[step,0]
            self.mmc.setPosition(self.mP['zDrive'], zFocusPos)
            self.mmc.waitForDevice(self.mP['zDrive'])
            self.snapImage()
       
            zFocus[step,0] = self.mmc.getPosition(self.mP['zDrive'])
            zFocus[step,1] = cv2.Laplacian(self.cameraImage, cv2.CV_64F ).var()
            zRecord[recInd,:2] = zFocus[step,:2]
            zRecord[recInd,2]  = zPhase
            recInd+=1

        zPhase  = 2
        maxInd = zFocus[:numSteps,1].argmax()
        ind = maxInd

        if ind == 0 or ind==numSteps: # Check to see if focus is at limits
            upZ  = zPos+zRange
            lowZ = zPos-zRange
            zFocus[:numSteps,0] = np.linspace(lowZ,upZ,numSteps, dtype = 'float')
            for step in range(numSteps):
                zFocusPos = zFocus[step,0]
                self.mmc.setPosition(self.mP['zDrive'], zFocusPos)
                self.mmc.waitForDevice(self.mP['zDrive'])
                self.snapImage()
               
                zFocus[step,0] = self.mmc.getPosition(self.mP['zDrive'])
                zFocus[step,1] = cv2.Laplacian(self.cameraImage,cv2.CV_64F ).var()
                zRecord[recInd,:2] = zFocus[step,:2]
                zRecord[recInd,2]  = zPhase
                
                recInd+=1
    
            maxInd = zFocus[:numSteps,1].argmax()
         
        zFocus[numSteps+2,:] = zFocus[maxInd,:]

        # Fine Tune Focus by sampleing between previous maximum points to find a
        # local maximum any spot that is low due to flickering of lamp will be
        # sorted out. 

        numIter = 1
       
        while (zIncrement>targetInc) & (numIter <= 20):
           
           # Calculate space between already measured top three maxima
            if maxInd == 0:
                zFocus[numSteps,0] =  zFocus[maxInd,0] - zIncrement
                stepVec = [numSteps]
                zPhase = 4.1
            
            elif maxInd  == numSteps: 
                zFocus[numSteps+1,0:2] =  zFocus[maxInd,0:2]
                zFocus[numSteps+2,0] =  zFocus[maxInd,0] + zIncrement
                stepVec = [numSteps+2]
                zPhase = 4.2
            
            else:
                zIncrement = zIncrement/2
                zFocus[numSteps,0] = zFocus[maxInd,0]-zIncrement
                zFocus[numSteps+2,0] = zFocus[maxInd,0]+zIncrement
                stepVec = [numSteps,numSteps+2]
                zPhase = zIncrement

            zFocusPos = zFocus[numSteps,0]-10
            
            self.mmc.setPosition(self.mP['zDrive'], zFocusPos) # This extra move takes out slack in the z-drive;
            self.mmc.waitForDevice(self.mP['zDrive'])
                 
            for step in stepVec:
                    
                zFocusPos = zFocus[step,0]
                
                self.mmc.setPosition(self.mP['zDrive'], zFocusPos)
                self.snapImage()
              
                zFocus[step,0] = self.mmc.getPosition(self.mP['zDrive'])
                zFocus[step,1] = cv2.Laplacian(self.cameraImage,cv2.CV_64F ).var()
                zRecord[recInd,:2] = zFocus[step,:2]
                zRecord[recInd,2] = zPhase
                recInd +=1

            maxInd = np.argmax(zFocus[:,1])

            zFocus[numSteps+1,:] = zFocus[maxInd,:]
            numIter +=1
        
       
        zFocusPos = zFocus[maxInd,0]
    
        # TODO:  Add plot for zFocus to evaluate 
        self.mmc.setPosition(self.mP['zDrive'],zFocusPos)
        self.mmc.waitForDevice(self.mP['zDrive'])
        
            
        if videoOn:
            self.videoTimer.start()

    def guiSetOrigin(self):

        self.xyzTime = np.array(([self.mmc.getXPosition(self.mP['xyDrive']), 
                                  self.mmc.getYPosition(self.mP['xyDrive']),
                                  self.mmc.getPosition(self.mP['zDrive']),
                                  np.datetime64(datetime.now()).astype('float')]),dtype = 'float') 

        self.mP['XYZOrigin'] = self.xyzTime[:3]
        self.updateControlPanel()

    def guiStartODELAY(self):

        videoOn      = self.controlPanel.focusButton.isChecked()
        videoRunning = self.videoTimer.isActive()
        odelayRunning = self.odelayTimer.isActive()

        if videoRunning:
            self.videoTimer.stop()

        # if not self.mP['experimentInitialized']:
        
        self.initializeExperiment()
        self.saveCurrentState()


        if not odelayRunning:
            self.odelayTimer.setInterval(self.mP['iterPeriod'])
            self.odelayTimer.timeout.connect(self.scanExp)
            self.odelayTimer.start()
            self.scanExp()
        
        else:
            self.odelayTimer.stop()
        
    def guiFocus(self):

        videoOn      = self.controlPanel.focusButton.isChecked()
        videoRunning = self.videoTimer.isActive()
        if videoRunning:
            self.videoTimer.stop()
        
        zCurPos = self.mmc.getPosition(self.mP['zDrive'])

        zRange = np.arange(-30, 30, 0.5)
        
        zFocusPos = np.tile(zCurPos, zRange.shape) + zRange

        zScore = np.zeros(zRange.shape, dtype = 'float')

        for n, zPos in enumerate(zFocusPos, 0):
            self.mmc.setPosition(self.mP['zDrive'], zPos)
            self.mmc.waitForDevice(self.mP['zDrive'])
            self.snapImage()

            zScore[n] = cv2.Laplacian(self.cameraImage, cv2.CV_64F).var()
        
        focusData = {}
        focusData['xData'] = zFocusPos
        focusData['yData'] = zScore

        self.focusPlot = FocusPlot(focusData)
        self.focusPlot.setGeometry(200,200,400,300)
        self.focusPlot.canvas.draw()
        self.focusPlot.show()

        
        return None

    def scanExp(self):
        '''
        1. Load microscope state from .config file and hdf5 data file
        2. Calculate Z Positions based on experiment format and last positions
        3. Set data log if active
        4. Loop through ROI list
        '''
        # self.mP = loadMicroscopeConfig()
        print('made it here')
        self.iterNum = self.mP['iterNum']

        self.focusPhase = 0
        if self.mP['iterNum'] >0:
            self.focusPhase = 1

        self.roiList = self.mP['roiList']
        # TODO:  needs roiOrder to arrange list

        for roi in self.roiList:
            self.roi = roi
            self.scanRoi(roi)
            
        self.mP['iterNum']+=1
        self.updateStageZPos()
        self.saveCurrentState()

        return None

    def saveCurrentState(self):

        microPrimaryPath = pathlib.Path.home() / '.microscopePaths'

        primaryMicroPropPath = pathlib.Path(self.primarySaveDir) / 'odelayExpConfig.cfg'
        primaryZStatePath    = pathlib.Path(self.primarySaveDir) / 'odelayZstate.hdf5'

        miroDirPaths = {'primaryMicroPath': str(primaryMicroPropPath)}

        with open(microPrimaryPath, 'w') as fileOut:
            json.dump(miroDirPaths, fileOut)

        with open(primaryMicroPropPath, 'w') as fileOut:
            json.dump(self.mP, fileOut)

        zState  = {}
        zState['zStagePos'] = self.zStagePos
        zState['zFocusPos'] = self.zFocusPos

        fio.saveDict(primaryZStatePath, zState) 

        return None
    
    def loadLastState(self):

        microPrimaryPath = pathlibPath.home() / '.microscopePaths'
        with open(microPrimaryPath, 'r') as fileIn:
            expDirectory = json.load(fileIn)

        self.primarySaveDir = expDirectory['primaryMicroPath']

        primaryMicroPropPath = pathlib.Path(self.primarySaveDir) / 'odelayExpConfig.cfg'
        primaryZStatePath    = pathlib.Path(self.primarySaveDir) / 'odelayZstate.hdf5'

        with open(primaryPropPath, 'r') as fileIn:
            self.mP = json.load(fileIn)

        zState = fio.loadData(primaryZStatePath)

        self.roiList = self.mP['roiList']
        self.iterNum = self.mP['iterNum']
        self.roi     = self.mP['roi']
        self.zStagePos = zState['zStagePos']
        self.zFocusPos = zState['zFocusPos']
        self.primarySaveDir = self.mP['primarySaveDir']
        self.backupSaveDir  = self.mP['backupSaveDir']

    def loadExperiment(self):
        

        return None

    def closeEvent(self, event):
        
        return None
    
if __name__ == '__main__':

    originDir = pathlib.Path.cwd()
    odelayPath = pathlib.Path(__file__).parent.parent
    os.chdir(odelayPath)

    app = QApplication(sys.argv)
    mastercontrol = MasterControl()
    sys.exit(app.exec_())
    