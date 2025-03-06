
import re
import os
import sys
import getpass
import pathlib
import time
import multiprocessing
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

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5              import QtCore, QtGui, QtWidgets
# from PyQt5.QtMultimedia import QMediaPlayer
# from PyQt5.QtMultimedia import QMediaContent
# from PyQt5.QtMultimediaWidgets import QVideoWidget
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

class StageMove(QGraphicsView):
    stageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(StageMove, self).__init__(parent)
    
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
        super(StageMove, self).mousePressEvent(event)

class StageFocus(QGraphicsView):
    focusClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(StageFocus, self).__init__(parent)
    
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
        super(StageFocus, self).mousePressEvent(event)

class StageNavigation(QGraphicsView):
    stageNavigationSignal = pyqtSignal(QPoint)

    def __init__(self, parent, mP):
        super(StageNavigation, self).__init__(parent)
    
        self.setMinimumSize(300,200)
        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap(QSize(mP['grid']['sceneDim'][0]/10, mP['grid']['sceneDim'][1]/10))
        self._pixmap.fill(QColor('blue'))
        
        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
    
        # Improve this section for experiment Properties
        wellDiameter = round(mP['grid']['wellDiameter']/10)
        xOffset = round(mP['grid']['wellSpacing']/10)
        yOffset = round(mP['grid']['wellSpacing']/10)
        roiList = [*mP['roiDict']]
        roiDict  = mP['roiDict']
        roiOrder = mP['roiOrder']
        
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
    
    def generateExpLayout(self, mP):

        self._scene  = QGraphicsScene(self)
        self._photo  = QGraphicsPixmapItem()
        self._pixmap = QPixmap(QSize(6700, 4100))
        self._pixmap.fill(QColor('blue'))
        
        self._photo.setPixmap(self._pixmap)
        self._scene.addItem(self._photo)
    
        # Improve this section for experiment Properties
        xOffset = round(mP['gridSpacing']/10)
        yOffset = round(mP['gridSpacing']/10)
        roiList = [*mP['roiDict']]
        roiDict  = mP['roiDict']
        roiOrder = mP['roiOrder']
        
        self.posMarker = []
        cntr = 0
        for roi in roiList:
            self.posMarker.append(PlatePosition(self))
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

        self.imageMarker = ImagePosition(self)
        self.imageMarker.setPos(450,450)
        self._scene.addItem(self.imageMarker)

        self.setScene(self._scene)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(255,255, 255,255)))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)
        
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
        self.createCubeSelect(mP)
        self.createCamera(mP)
        self.createIllumination(mP)
        self.createObjective(mP)
        self.createFocusTest()
       
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.stageControl,  0, 0, 2, 2)
        mainLayout.addWidget(self.focusControl,  0, 8, 2, 2)
        mainLayout.addWidget(self.cubeSelect,    6, 0, 4, 8)
        mainLayout.addWidget(self.cameraControl, 6, 8, 4, 1)
        mainLayout.addWidget(self.illumControl,  0, 6, 2, 2)
        mainLayout.addWidget(self.objectiveSelect,  3, 6, 1, 2)
        mainLayout.addWidget(self.focusTest,        3, 8, 1, 1)

        self.setLayout(mainLayout)

        self.setWindowTitle("Microscope Control")

    def createStageControl(self, mP):

        self.stageControl  = QGroupBox('Stage Control')
        self.stageLayout   = QGridLayout()
        self.stageMove     = StageMove(self)
        self.stageMove.stageClicked.connect(self.stageClicked)
        self.stageFocus    = StageFocus(self)
        self.stageFocus.focusClicked.connect(self.focusClicked)

        self.stageNavigate = StageNavigation(self, mP)
        self.stageNavigate.stageNavigationSignal.connect(self.navClicked)
       
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
        self.stageLayout.addWidget(self.startODELAY,    1,0,1,1)

        self.stageLayout.addWidget(self.navPixInfo,     4,0,1,1)
        self.stageLayout.addWidget(self.xOrigin,        1,1,1,1)
        self.stageLayout.addWidget(self.yOrigin,        2,1,1,1)
        self.stageLayout.addWidget(self.zOrigin,        3,1,1,1)

        self.stageLayout.addWidget(self.setOrigin,      4,1,1,1)
        self.stageLayout.addWidget(self.measureDist,    4,2,1,1)
        
        self.stageLayout.addWidget(self.xStagePos,      1,2,1,1)
        self.stageLayout.addWidget(self.yStagePos,      2,2,1,1)
        self.stageLayout.addWidget(self.zStagePos,      3,2,1,1)
        
        self.stageLayout.addWidget(self.stageNavigate,  0,5,5,5)
        self.stageControl.setLayout(self.stageLayout)

    def createCubeSelect(self, mP):

        self.cubeSelect = QGroupBox('Cube Select')
        cubeLayout = QGridLayout()
        # Define buttons and widgets in box
        cubeList = mP['imageModeList']
        cubeDict = mP['cubeDict']

        self.label  = {}
        self.expTime = {}
        self.selectCubeButton = {}
        self.recordCubeButton = {}
    
        for n, cubeID in enumerate(cubeList,0):
            self.label[cubeID] = QLabel(f'{cubeID}-{n}')
            self.expTime[cubeID]   = QLineEdit()
            self.expTime[cubeID].setMinimumSize(QSize(80, 20))
            self.expTime[cubeID].setAlignment(Qt.AlignHCenter) 
            self.expTime[cubeID].setText(str(cubeDict[cubeID][2]))

            self.selectCubeButton[cubeID] = QPushButton()
            self.selectCubeButton[cubeID].setText(cubeID)
            self.selectCubeButton[cubeID].setObjectName(f'selectButton_{cubeID}')
            self.selectCubeButton[cubeID].setMinimumSize(QSize(95,150))
            self.recordCubeButton[cubeID] = QPushButton()
            self.recordCubeButton[cubeID].setText(cubeID)
            self.recordCubeButton[cubeID].setObjectName(f'recordButton_{cubeID}')
            self.recordCubeButton[cubeID].setMinimumSize(QSize(40,20))

            cubeWL = mP['cubeDict'][cubeID][0]
            rgb = odp.waveLengthToRGB(cubeWL)

            self.selectCubeButton[cubeID].setCheckable(True)
            self.recordCubeButton[cubeID].setCheckable(True)
            qss= f'QPushButton:checked {{background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); font-size: 10pt; font-weight: bold}}\
                   QPushButton{{background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); font-size: 10pt; font-weight: bold}}'

            self.selectCubeButton[cubeID].setStyleSheet(qss)
            self.recordCubeButton[cubeID].setStyleSheet(qss)
            
            cubeLayout.addWidget(self.label[cubeID],            0, n + 1, 1, 1, Qt.AlignCenter)
            cubeLayout.addWidget(self.expTime[cubeID],          1, n + 1, 1, 1, Qt.AlignCenter)
            cubeLayout.addWidget(self.selectCubeButton[cubeID], 2, n + 1, 3, 1, Qt.AlignCenter)
            cubeLayout.addWidget(self.recordCubeButton[cubeID], 5, n + 1, 1, 1, Qt.AlignCenter)

        self.cubeSelect.setLayout(cubeLayout)
        # breakpoint()

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
        self.autoFocus_1  = QPushButton()
        self.autoFocus_1.setObjectName('autoFocus_1')
        self.autoFocus_1.setText('Auto Focus 2')
        self.focusRange_0 = QLineEdit()
        self.focusRange_0.setText(str(mP['AutoFocus_0']['zRange']))
        self.focusRange_1 = QLineEdit()
        self.focusRange_1.setText(str(mP['AutoFocus_1']['zRange']))
        self.numSteps_0   = QLineEdit()
        self.numSteps_0.setText(str(mP['AutoFocus_0']['numSteps']))
        self.numSteps_1   = QLineEdit()
        self.numSteps_1.setText(str(mP['AutoFocus_1']['numSteps']))
        self.targetInc_0  = QLineEdit()
        self.targetInc_0.setText(str(mP['AutoFocus_0']['targetInc']))
        self.targetInc_1  = QLineEdit()
        self.targetInc_1.setText(str(mP['AutoFocus_1']['targetInc']))

        autofocusLayout.addWidget(self.autoFocus_0,  0, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.autoFocus_1,  0, 1, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.focusRange_0, 1, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.focusRange_1, 1, 1, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.numSteps_0,   2, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.numSteps_1,   2, 1, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.targetInc_0,  3, 0, 1, 1, Qt.AlignCenter)
        autofocusLayout.addWidget(self.targetInc_1,  3, 1, 1, 1, Qt.AlignCenter)

        self.focusControl.setLayout(autofocusLayout)

    def createCamera(self, mP):
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

    def createObjective(self, mP):

        self.objectiveSelect = QGroupBox('Objective')
        objectiveLayout = QGridLayout()
        objectiveList = mP['configDict']['Objective']
        
        self.selectObjButton = {}
    
        for n, objID in enumerate(objectiveList,0):
    
            self.selectObjButton[objID] = QPushButton()
            self.selectObjButton[objID].setText(objID)
            self.selectObjButton[objID].setObjectName(f'selectObjective_{objID}')
            self.selectObjButton[objID].setCheckable(True)
            qss= f'QPushButton:checked {{background-color:rgb(0,0,255); font-size: 10pt; font-weight: bold}}\
                   QPushButton{{background-color:rgb(200,200,200); font-size: 10pt; font-weight: bold}}'

            self.selectObjButton[objID].setStyleSheet(qss)
            objectiveLayout.addWidget(self.selectObjButton[objID], 0, n + 1, 1, 1)

        self.objectiveSelect.setLayout(objectiveLayout)

    def createFocusTest(self):
        self.focusTest = QGroupBox('Test')
        focusTestLayout = QGridLayout()

        self.focusTestButton = QPushButton()
        self.focusTestButton.setText('Plot Focus')
        focusTestLayout.addWidget(self.focusTestButton)
        self.focusTest.setLayout(focusTestLayout)

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

class DisplayPanel(QWidget):
    
    def __init__(self):
        super(DisplayPanel, self).__init__()
          
        # TODO: bring in mP to set initial sensor size
        self.imageData = np.zeros((2048,2048), dtype = 'uint16')
        
        mainLayout = QGridLayout()

        self.imageDisplay = ImageDisplay(self)
        self.histDisplay = HistogramDisplay(self)

        mainLayout.addWidget(self.imageDisplay, 0,0)
        mainLayout.addWidget(self.histDisplay,  0,1)

        self.setLayout(mainLayout)
        self.setWindowTitle("Image Display")

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

class ImageDisplay(QGraphicsView):
    # TODO:  Add in zoom and plotting for reviewing data.
    def __init__(self, parent):
        super(ImageDisplay, self).__init__(parent)

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

class RoiSaveProcess(multiprocessing.Process):

    def __init__(self, filePaths, imageRoi, mP, zState):  
        super(RoiSaveProcess, self).__init__()  
        self.filePaths = filePaths 
        self.imageRoi  = imageRoi
        self.mP        = mP
        self.zState    = zState

    def run(self): 
        # try:
        filePath = self.filePaths['primary path']
        
        microPrimaryPath = pathlib.Path.home() / '.microscopePaths'
        primaryMicroPropPath = pathlib.Path(self.mP['primarySaveDir']) / 'odelayExpConfig.cfg'
        primaryZStatePath    = pathlib.Path(self.mP['primarySaveDir']) / 'odelayZstate.hdf5'


        with open(primaryMicroPropPath, 'w') as fileOut:
            json.dump(self.mP, fileOut)
        fio.saveDict(primaryZStatePath, self.zState) 
        fio.saveDict(filePath, self.imageRoi)
           
        # except:
        #     backupPath = self.filePaths['backup path']
        #     backupMicroPropPath = pathlib.Path(self.mP['primarySaveDir']) / 'odelayExpConfig.cfg'
        #     backupZStatePath    = pathlib.Path(self.mP['primarySaveDir']) / 'odelayZstate.hdf5'

        #     with open(backupMicroPropPath, 'w') as fileOut:
        #         json.dump(self.mP, fileOut) 
        #     fio.saveDict(backupZStatePath, zState) 
        #     fio.saveDict(backupPath, self.imageRoi)
        #     fio.saveDict(backupPath, self.imageRoi)

        return None

class MasterControl(QMainWindow):
    '''
    MasterControl is the base window for our controller
    '''

    def __init__(self):
        super(MasterControl, self).__init__()

        # TODO: Create window with 
        # self.createMenus()
        self.primarySaveDir = None
        self.backupSaveDir  = None
        self.microscopeConfigFilePath = None
        self.autoFocusParams = {}

        self.setWindowTitle('ODELAY!!!!')

        self.text_ExperimentName = QLabel()
        self.text_ExperimentName.setText('Experiment Name')
        self.ledit_ExperimentName   = QLineEdit()

        self.text_PrimaryDir = QLabel()
        self.text_PrimaryDir.setText('Primary Directory')

        self.ledit_PrimaryDirectory   = QLineEdit()
        self.button_ChoosePrimaryDir  = QPushButton()
        self.button_ChoosePrimaryDir.setText('Primary Dir')
        self.button_ChoosePrimaryDir.clicked.connect(self.choosePrimaryDir)

        self.text_BackupDir = QLabel()
        self.text_BackupDir.setText('Backup Directory')
        self.ledit_BackupDirectory  = QLineEdit()

        self.button_ChooseBackupDir = QPushButton()
        self.button_ChooseBackupDir.setText('Backup Dir')
        self.button_ChooseBackupDir.clicked.connect(self.chooseBackupDir)

        self.text_ExperimentType = QLabel()
        self.text_ExperimentType.setText('Experimet Type')
        self.selectExperimentType   = QComboBox()
        experimentList = ['ODELAY 96 Spot', 'ODELAY 5 Condition', 'Macrophage 24 well', 'External File']
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
        masterGridLayout.addWidget(self.text_ExperimentName,     0,0,1,1)
        masterGridLayout.addWidget(self.ledit_ExperimentName,    0,1,1,1)

        masterGridLayout.addWidget(self.text_PrimaryDir,         1,0,1,1)
        masterGridLayout.addWidget(self.ledit_PrimaryDirectory,  1,1,1,1)
        masterGridLayout.addWidget(self.button_ChoosePrimaryDir, 1,2,1,1) 

        masterGridLayout.addWidget(self.text_BackupDir,          2,0,1,1)
        masterGridLayout.addWidget(self.ledit_BackupDirectory,   2,1,1,1)
        masterGridLayout.addWidget(self.button_ChooseBackupDir,  2,2,1,1) 

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

        self.primarySaveDir = pathlib.Path('L:\ODELAY HPC Temp\ODELAY 2020-06-27 Nup170 MMS')
        self.backupSaveDir  = pathlib.Path('L:\\ODELAY HPC Temp\\ODELAY TestFile\\BackUp')
        # self.microscopeConfigFilePath =  pathlib.Path('N:\Python-Projects\odelay-EWRWLK176\MMConfig_LeicaDMI_EWRWLK176.cfg')
        self.microscopeConfigFilePath =  pathlib.Path('C:\\Program Files\\Micro-Manager-2.0gamma\\MMConfig_demo.cfg')

        self.selectExperiment('ODELAY 96 Spot')

        self.createInterface()

    def selectExperiment(self, modName = None):
        if modName == None:
            modName = self.selectExperimentType.currentText()

        mP = self.microscopeDefaultConfig() 
        expDict = expMods.moduleSelector(modName, mP)

        self.mP = expDict

        return None

    def choosePrimaryDir(self):

        options = QFileDialog.Options()
        dirName = QFileDialog.getExistingDirectory(self,"Select ODELAY Primary Directory", "", options=options)

        if len(dirName)>0:
            primarySavePath = pathlib.Path(dirName)
            self.ledit_PrimaryDirectory.setText(str(primarySavePath))
            self.primarySaveDir = primarySavePath
            if self.backupSaveDir == None:
                qss= f'QPushButton{{background-color:rgb(255,255,0); font-size: 10pt; font-weight: bold}}'

        return None

    def chooseBackupDir(self):

        options = QFileDialog.Options()
        dirName = QFileDialog.getExistingDirectory(self,"Select ODELAY Backup Directory", "", options=options)

        if len(dirName)>0:
            backupSavePath = pathlib.Path(dirName)
            self.ledit_BackupDirectory.setText(str(backupSavePath))
            self.backupSaveDir = backupSavePath

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

            self.mmc = self.connectMicroscope()
  
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
            self.controlPanel.setGeometry(500, 200, 800, 600)
            self.controlPanel.show()
            self.controlPanel.navUpdate()

            self.displayPanel = DisplayPanel()
            self.displayPanel.setGeometry(100, 100, 800, 800)
            self.displayPanel.show()

            # Connect Signals to Slots here for ordered control of instrument.

            self.controlPanel.focusButton.clicked.connect(self.videoControl)
            self.controlPanel.recordButton.clicked.connect(self.recordImage)

            self.controlPanel.autoFocus_0.clicked.connect(self.autoFocusButton)
            self.controlPanel.autoFocus_1.clicked.connect(self.autoFocusButton)

            self.controlPanel.stageMove.stageClicked.connect(self.guiMoveXYStage)
            self.controlPanel.stageFocus.focusClicked.connect(self.guiMoveZStage)

            self.controlPanel.startODELAY.clicked.connect(self.guiStartODELAY)
            self.controlPanel.setOrigin.clicked.connect(self.guiSetOrigin)

            for roiInd in range(len(self.controlPanel.stageNavigate.posMarker)):
                self.controlPanel.stageNavigate.posMarker[roiInd].call_Position.connect(self.navMoveStage)

            for cubeID in self.controlPanel.selectCubeButton:
                self.controlPanel.selectCubeButton[cubeID].clicked.connect(self.selectCubePos)
                
            for cubeID in self.controlPanel.recordCubeButton:
                self.controlPanel.recordCubeButton[cubeID].clicked.connect(self.selectRecord)

            for objID in self.controlPanel.selectObjButton:
                self.controlPanel.selectObjButton[objID].clicked.connect(self.selectObjectiveLens)

            if 'Bf' in [*self.controlPanel.selectCubeButton]:
                self.controlPanel.recordCubeButton['Bf'].setChecked(True)
                self.controlPanel.selectCubeButton['Bf'].setChecked(True)
                exposeTime = int(self.controlPanel.expTime['Bf'].text())
                microState = 'Bf'
                self.mmc.setExposure(exposeTime)
                self.mmc.setConfig('Channel', microState)
                
                self.controlPanel.selectObjButton['10x'].setChecked(True)
                self.mmc.setConfig('Objective', '10x')

            self.controlPanel.focusTestButton.clicked.connect(self.generateFocusCurve)
            self.snapImage()
    
    def openConfigFileDialog(self):
    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,"Select Microscope Config File", "","Micro-M  (*.cfg);;", options=options)
        return fileName

    def saveDirectoryDialog(self):
    
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getExistingDirectory(self,"Select Image Directory", "","", options=options)
        return fileName

    def connectMicroscope(self):

        if self.microscopeConfigFilePath == None:

            system_cfg_file = 'C:\\Program Files\\Micro-Manager-2.0gamma\\MMConfig_demo.cfg'
        else:
            system_cfg_file = str(self.microscopeConfigFilePath)

        # For most devices it is unnecessary to change to the MM direcotry prior to importing, but in some cases (such as the pco.de driver), it is required.
        # breakpoint()
        prev_dir = os.getcwd()
        try:
            sys.path.append('C:\\Program Files\\Micro-Manager-2.0gamma')
            os.chdir('C:\\Program Files\\Micro-Manager-2.0gamma')

        except:
            sys.path.append('C:\\Micro-Manager-1.4')
            os.chdir('C:\\Micro-Manager-1.4')
        
        import MMCorePy

        # Get micro-manager controller object
        mmc = MMCorePy.CMMCore()

        # Load system configuration (loads all devices)
        mmc.loadSystemConfiguration(system_cfg_file)
        os.chdir(prev_dir)

        # Success!
        print("Micro-manager was loaded sucessfully!")

        return mmc

    def microscopeDefaultConfig(self):
        mP = {}

        #  AutoFocus Props
        mP['odelayStarted'] = False
        mP['experimentInitialized'] = False

        mP['AutoFocus_0'] = {}
        mP['AutoFocus_0']['zRange']    = 80
        mP['AutoFocus_0']['numSteps']  = 11
        mP['AutoFocus_0']['targetInc'] = 0.3

        mP['AutoFocus_1'] = {}
        mP['AutoFocus_1']['zRange']    = 20
        mP['AutoFocus_1']['numSteps']  = 7
        mP['AutoFocus_1']['targetInc'] = 0.3

        mP['grid'] = {}
        mP['grid']['rowName'] = ['E','F','G','H','I','J','K','L']
        mP['grid']['colName'] = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19']
        mP['grid']['xgrid'] = np.arange(0,4500*14,4500,dtype='float').tolist()
        mP['grid']['ygrid'] = np.arange(0,4500*8,4500, dtype='float').tolist()

        mP['numTimePoints'] = 96
        mP['numWells'] = len(mP['grid']['rowName'])*len(mP['grid']['colName']) # convert to method

        mP['stageXYPos'] = np.zeros((mP['numWells'], 2),dtype = 'float').tolist()

        cnt = 0
        stageXYPos = {}
        roiDict    = {}
        for row, rowID in enumerate(mP['grid']['rowName'],0):
            for col, colID in enumerate(mP['grid']['colName'],0):
          
                stageXYPos[f'{rowID}{colID}']       = [mP['grid']['xgrid'][col], mP['grid']['ygrid'][row], cnt]
                roiDict[f'{rowID}{colID}'] = {}
                roiDict[f'{rowID}{colID}']['xyPos'] = [mP['grid']['xgrid'][col], mP['grid']['ygrid'][row]]
                roiDict[f'{rowID}{colID}']['ImageConds'] = {'Bf': 5}
                cnt+=1
        mP['stageXYPos'] = stageXYPos
        mP['roiDict']    = roiDict
        
        mP['iterNum']  = 0
        mP['roiIndex'] = 0
        mP['roi'] = mP['roiIndex']

        mP['totIter']      = 96
        mP['iterPeriod']   = 1800*1000
        mP['getFileFlag']  = True
        mP['ErrorOccured'] = False
        mP['restartCnt']   = 0

        mP['numTiles']   = 9
        
        mP['overlapPerc'] = 0.2
        mP['pixSize']  = 6.5
        mP['tileOrder']= [[-1, -1],
                          [ 0, -1],
                          [ 1, -1],
                          [-1,  0], 
                          [ 0,  0],
                          [ 1,  0],
                          [-1,  1],
                          [ 0,  1],
                          [ 1,  1]]

        #  Microscope Conditions              
        mP['XYZOrigin'] = [35087,26180 , 6919];      
     
        cubeDict = {}   
        cubeDict['Cy5']         = [647, 0, 50]
        cubeDict['DAPI']        = [461, 0, 50]
        cubeDict['FITC']        = [519, 0, 50]
        cubeDict['GFPHQ']       = [519, 0, 50]
        cubeDict['TxRed']       = [519, 0, 50]
        cubeDict['Blue']        = [519, 0, 50]
        cubeDict['AT-AQ']       = [519, 0, 50]
        cubeDict['RHOD']        = [552, 0, 50]
        cubeDict['525']         = [525, 0, 50]
        cubeDict['565']         = [565, 1, 50]
        cubeDict['605']         = [605, 2, 50]
        cubeDict['655']         = [655, 3, 50]
        cubeDict['705']         = [705, 4, 50]
        cubeDict['ATL']         = [425, 0, 50]
        cubeDict['642']         = [642, 1, 50]
        cubeDict['TXR']         = [605, 2, 50]
        cubeDict['A4']          = [655, 3, 50]
        cubeDict['Bf']          = [425, 5, 5]
        cubeDict['BrightField'] = [425, 5, 5]
        cubeDict['DarkField']   = [425, 5, 5]
        mP['cubeDict']          = cubeDict
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
        if filePath == None:
            filePath = pathlib.Path(__file__).parent
        
        if not 'mP' in dir(self):
            self.selectExperiment()

        microscopeConfigPath =  [mPath for mPath in filePath.glob('*.mmconfig')]

        configGroups = self.mmc.getAvailableConfigGroups()      
        imageModeList = []
        configList = []
        configDict = {}

        for group in configGroups:
            configList.append(group)
            groupList = self.mmc.getAvailableConfigs(group)
            configDict[group] = [mode for mode in groupList]
            for mode in groupList:
                imageModeList.append(mode)

        self.mP['primarySaveDir']           = str(self.primarySaveDir)
        self.mP['backupSaveDir']            = str(self.backupSaveDir)
        self.mP['microscopeConfigFilePath'] = str(self.microscopeConfigFilePath) 

        self.mP['xyDrive'] = self.mmc.getXYStageDevice()
        self.mP['zDrive']  = self.mmc.getFocusDevice()
        self.mP['shutter'] = self.mmc.getShutterDevice()
        self.mP['imageModeList'] = configDict['Channel']
        self.mP['configDict']    = configDict
        self.mP['configList']    = 'Channel'

        if self.mP['xyDrive'] == 'TIXYDrive':
            self.xyzDir = np.array([-1,1,1], dtype='float')
            self.mP['XYZOrigin']=[29500,-15500,8500.0]

        elif self.mP['xyDrive'] == 'XY':
            self.xyzDir = np.array([1,1,1], dtype='float')
            self.mP['XYZOrigin']=[0,0,0]

        else:
            self.xyzDir = np.array([1,1,1], dtype='float')
        
        absXPos = self.mmc.getXPosition(self.mP['xyDrive'])
        absYPos = self.mmc.getYPosition(self.mP['xyDrive'])
        absZPos = self.mmc.getPosition( self.mP['zDrive'])

        self.absXYZPos = np.array([absXPos, absYPos, absZPos], dtype = 'float')
        self.relXYZPos = self.absXYZPos*self.xyzDir + np.array(self.mP['XYZOrigin'], dtype = 'float')

        return None

    def loadExcelMicroscopeConfig(self, filePath):

        configDict = fio.readMMConfigFile(filePath)

    def calculateTilePostions(self):

        sensorSize = np.array(self.cameraImage.shape, dtype='float')
        overlap    = self.mP['overlapPerc']
        pixSize    = self.mmc.getPixelSizeUm()        
        tileOrder  = np.array((self.mP['tileOrder']), dtype='float')
        numTiles = tileOrder.shape[0]
        # tilePos = tileOrder*np.tile(sensorSize,(numTiles,1))*pixSize/mag(1-overlap)

        self.tilePos = tileOrder*np.tile(sensorSize,(numTiles,1))*pixSize*(1-overlap)
        # print(self.tilePos)

        return None

    def videoControl(self):
        
        videoOn         = self.controlPanel.focusButton.isChecked()
        recordOn        = self.controlPanel.recordButton.isChecked()
        odelayRecording = self.odelayTimer.isActive()

        # self.mmc.setExposure(exposeTime)
        # self.mmc.setConfig('Channel', microState)
        # self.mmc.waitForSystem()
        # self.videoTimer.setInterval(50)

        if videoOn and not odelayRecording:
            timerDuration = self.mmc.getExposure() + 25
            self.videoTimer.setInterval(timerDuration)
            self.videoTimer.start()
            self.controlPanel.recordButton.setChecked(False)
        else:
            self.videoTimer.stop()
            self.controlPanel.focusButton.setChecked(False)
            self.controlPanel.recordButton.setChecked(False)

        return None

    def snapImage(self):

        self.mmc.snapImage()
        self.cameraImage = self.mmc.getImage()
        self.xyzTime = np.array(([self.mmc.getXPosition(self.mP['xyDrive']), 
                                  self.mmc.getYPosition(self.mP['xyDrive']),
                                  self.mmc.getPosition( self.mP['zDrive']),
                                  np.datetime64(datetime.now()).astype('float')]),dtype = 'float') 

        if self.cameraImage.shape[0]==2048:
            self.displayPanel.imageUpdate(cv2.resize(self.cameraImage, (512,512)))
        else:
            self.displayPanel.imageUpdate(self.cameraImage)

        
        self.updateControlPanel()
        QApplication.processEvents()

    def recordImage(self):
        videoOn         = self.controlPanel.focusButton.isChecked()
        odelayRecording = self.odelayTimer.isActive()

        if videoOn and not odelayRecording:
            self.videoTimer.stop()
            self.controlPanel.focusButton.setChecked(False)

        if odelayRecording:
            imageConds = self.mP['roiDict'][self.roi]['ImageConds']                
        else:
            imageConds = {'Bf': 5}
            for cubeID in self.controlPanel.recordCubeButton:
                if self.controlPanel.recordCubeButton[cubeID].isChecked():
                    imageConds[cubeID]  = int(self.controlPanel.expTime[cubeID].text())
        
        self.imageStack = {}

        for microState, exposeTime in imageConds.items():
           
            self.mmc.setExposure(exposeTime)
            self.mmc.setConfig('Channel', microState)
            self.mmc.waitForSystem()
            self.snapImage() 
            self.imageStack[microState] = {}
            self.imageStack[microState]['image']   = self.cameraImage
            self.imageStack[microState]['xyzTime'] = self.xyzTime

        self.controlPanel.recordButton.setChecked(False)
        if videoOn and not odelayRecording:
            self.videoTimer.start()

    def navMoveStage(self, posIndex):
        roiList = [*self.mP['roiDict']]
        roiDict  = self.mP['roiDict']
        roiOrder = self.mP['roiOrder']

        xyPos = roiDict[roiList[posIndex]]['xyPos']
        newXYPos = np.array(xyPos, dtype = 'float') + np.array(self.mP['XYZOrigin'][:2], dtype = 'float')
       
        while self.mmc.deviceBusy(self.mP['xyDrive']):
            time.sleep(0.2)
    
        self.mmc.setXYPosition(self.mP['xyDrive'],newXYPos[0], newXYPos[1])
    
    def guiMoveXYStage(self,movePos):

        absXYPos = self.mmc.getXYPosition(self.mP['xyDrive'])
       
        xOrg = self.controlPanel.stageMove._pixmap.width()/2
        yOrg = self.controlPanel.stageMove._pixmap.height()/2

        maxXYMove = 100
        dispX = maxXYMove*(movePos.x()-xOrg)/xOrg
        dispY = maxXYMove*(movePos.y()-yOrg)/yOrg
        
        newXYPos = np.array((dispX, dispY), dtype = 'float') + absXYPos
        self.controlPanel.navPixInfo.setText('%d, %d' % (newXYPos[0], newXYPos[1]))      

        self.mmc.setXYPosition(self.mP['xyDrive'], newXYPos[0], newXYPos[1])

        return None

    def guiMoveZStage(self,Pos):
        
        # get Current Z-position
        zPos = self.mmc.getPosition(self.mP['zDrive'])

        yOrigin = self.controlPanel.stageFocus._pixmap.height()/2
        maxZMove = 10
        dispZ = maxZMove*(yOrigin - Pos.y())/yOrigin
        newZPos = dispZ+zPos
        self.mmc.setPosition(self.mP['zDrive'], newZPos)
        self.controlPanel.navPixInfo.setText('%d' % newZPos)
        self.updateControlPanel()

        return None

    def moveXYStage(self, xPos, yPos):
        '''
        Handles moving the stage XY positions.   
        '''
        xyPos = np.array([xPos,yPos], 'float')
        xyDir = self.xyzDir[:2]
        xyOrigin = self.mP['XYZOrigin'][:2]

        newXYPos = xyPos*xyDir + xyOrigin

        self.mmc.setXYPosition(self.mP['xyDrive'],newXYPos[0], newXYPos[1])
        self.mmc.waitForDevice(self.mP['xyDrive'])

        absXPos = self.mmc.getXPosition(self.mP['xyDrive'])
        absYPos = self.mmc.getYPosition(self.mP['xyDrive'])
        absZPos = self.mmc.getPosition(self.mP['zDrive'])

        self.absXYZPos = np.array([absXPos, absYPos, absZPos], dtype = 'float')
        self.relXYZPos = self.absXYZPos*self.xyzDir + np.array(self.mP['XYZOrigin'], dtype = 'float')

        return None
    
    def moveZStage(self, zPos):
        
        newZPos = zPos + self.mP['XYZOrigin'][2]
        self.mmc.setPosition(self.mP['zDrive'], newZPos)
        self.mmc.waitForDevice(self.mP['zDrive'])

        absXPos = self.mmc.getXPosition(self.mP['xyDrive'])
        absYPos = self.mmc.getYPosition(self.mP['xyDrive'])
        absZPos = self.mmc.getPosition(self.mP['zDrive'])

        self.absXYZPos = np.array([absXPos, absYPos, absZPos], dtype = 'float')
        self.relXYZPos = self.absXYZPos - self.mP['XYZOrigin']

        return None

    def transShutter(self):

        shutterOpen = self.controlPanel.transShutter.isChecked()
        if shutterOpen:
            self.mmc.setProperty(self.mP['shutter'], 'State', 1)
        else:
            self.mmc.setProperty(self.mP['shutter'], 'State', 0)
    
        return None
    
    def fluorShutter(self):

        shutterOpen = self.controlPanel.fluorShutter.isChecked()
        if shutterOpen:
            self.mmc.setProperty(self.mP['shutter'], 'State', 1)
        else:
            self.mmc.setProperty(self.mP['shutter'], 'State', 0)
    
        return None

    def selectRecord(self):

        pass

    def selectCubePos(self):

        cubeSender = self.sender()
        cubeList = [*self.controlPanel.selectCubeButton]
        cubeSelected = re.sub("selectButton_","", cubeSender.objectName())
        cubeList.remove(cubeSelected)
        
        exposeTime = int(self.controlPanel.expTime[cubeSelected].text())
        configGroup = self.mP['configList']

        self.mmc.setExposure(exposeTime)
        self.mmc.setConfig(configGroup,cubeSelected)
        
        self.controlPanel.selectCubeButton[cubeSelected].setChecked(True)

        for cube in cubeList:
            self.controlPanel.selectCubeButton[cube].setChecked(False)        

    def selectObjectiveLens(self):
        objSender = self.sender()
        objList   = [*self.controlPanel.selectObjButton]
        objSelected = re.sub("selectObjective_","", objSender.objectName())
        objList.remove(objSelected)

        self.mmc.setConfig('Objective',objSelected)
        
        self.controlPanel.selectObjButton[objSelected].setChecked(True)

        for objID in objList:
            self.controlPanel.selectObjButton[objID].setChecked(False)        

    def updateControlPanel(self):
    
        self.absXYZPos = np.array(([self.mmc.getXPosition(self.mP['xyDrive']), 
                                    self.mmc.getYPosition(self.mP['xyDrive']),
                                    self.mmc.getPosition(self.mP['zDrive'])]),dtype = 'float') 
        
        self.relXYZPos = self.absXYZPos*self.xyzDir - np.array(self.mP['XYZOrigin'], dtype = 'float')

        xyz  = np.round(self.relXYZPos,decimals = 2)  
        xyzO = np.round(self.mP['XYZOrigin'], decimals = 2)
        
        self.controlPanel.xOrigin.setText(str(xyzO[0])) 
        self.controlPanel.yOrigin.setText(str(xyzO[1])) 
        self.controlPanel.zOrigin.setText(str(xyzO[2]))

        self.controlPanel.xStagePos.setText(str(xyz[0])) 
        self.controlPanel.yStagePos.setText(str(xyz[1]))
        self.controlPanel.zStagePos.setText(str(xyz[2]))

        xOffset = round(self.mP['grid']['wellSpacing']/10)
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

        elif focusPhase == 2:
            self.autoFocusParams['zRange']   = int(8)
            self.autoFocusParams['numSteps'] = int(5)
            self.autoFocusParams['targetInc']= float(0.3)        
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
         
        zFocus[numSteps+1,:] = zFocus[maxInd,:]

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
            
            elif maxInd  == numSteps-1:
                zFocus[numSteps,0] =  zFocus[maxInd,0] - zIncrement 
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
            # maxInd = maxInds
            zFocus[numSteps+1,:] = zFocus[maxInd,:]
            numIter +=1
        
       
        zFocusPos = zFocus[maxInd,0]
        # print(zFocusPos)
        # TODO:  Add plot for zFocus to evaluate 
        self.mmc.setPosition(self.mP['zDrive'],zFocusPos)
        # self.mmc.waitForDevice(self.mP['zDrive'])
        
            
        if videoOn:
            self.videoTimer.start()

    def guiSetOrigin(self):

        self.xyzTime = np.array(([self.mmc.getXPosition(self.mP['xyDrive']), 
                                  self.mmc.getYPosition(self.mP['xyDrive']),
                                  self.mmc.getPosition(self.mP['zDrive']),
                                  np.datetime64(datetime.now()).astype('float')]),dtype = 'float') 

        self.mP['XYZOrigin'] = self.xyzTime[:3].tolist()
        self.updateControlPanel()

    def guiStartODELAY(self):

        videoOn      = self.controlPanel.focusButton.isChecked()
        videoRunning = self.videoTimer.isActive()
        odelayRunning = self.odelayTimer.isActive()

        if videoRunning:
            self.videoTimer.stop()

        if not self.mP['experimentInitialized']:
            self.initializeExperiment()
            self.saveCurrentState()
        
        if not odelayRunning and (self.mP['iterPeriod'] !=0):
            if not self.mP['odelayStarted']:
                self.odelayTimer.setInterval(self.mP['iterPeriod'])
                self.odelayTimer.timeout.connect(self.scanExp)
                self.mP['odelayStarted'] = True

            print('Runing on Timer')
            self.odelayTimer.start()
            self.scanExp()
            
        elif self.mP['iterPeriod'] == 0:
            while (self.mP['iterNum'] <= self.mP['numTimePoints']) and self.controlPanel.startODELAY.isChecked():
                print(f'Not Running on Timer attempting to loop {self.iterNum}')
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
       
        self.iterNum = self.mP['iterNum']
        

        self.focusPhase = 0
        if self.mP['iterNum'] > 0:
            self.focusPhase = 1
        self.roiList = self.mP['roiList']
        # TODO:  needs roiOrder to arrange list

        for roi in self.roiList:
            self.roi = roi
            self.scanRoi(roi)

        self.mP['iterNum']+=1
       
        self.updateStageZPos()
        # self.saveCurrentState()

        
        return None

    def scanRoi(self, roi):
        '''
        Scan region of interest loading each configuration state for recording images.  
        This method does not currently support z-stacks but could be added in a number 
        of ways.  Most likely add an additioanal dimention to tilePos to include z-positions
        '''
        print(f'moving to {roi}')
        self.calculateTilePostions()
        numTiles = self.mP['numTiles']
        roiList  = self.mP['roiList']
        roiOrder = self.mP['roiOrder']

        stageXYPos = np.array(self.mP['stageXYPos'][roi], dtype = 'float')
        ind = roiList.index(roi)
        stageZPos  = self.zStagePos[ind, self.iterNum]
        scanPos    = np.tile(stageXYPos[:2], (numTiles,1)) + self.tilePos 

        self.imageConds = self.mP['roiDict'][roi]['ImageConds']

        self.moveXYStage(stageXYPos[0],stageXYPos[1])
        self.mmc.setPosition(self.mP['zDrive'], float(stageZPos))
        
        self.setautoFocusParams(self.focusPhase)
        self.autoFocus()

        # zPos = self.mmc.getPosition(self.mP['zDrive'])-5
        # self.mmc.setPosition(self.mP['zDrive'], float(zPos))
        # self.setautoFocusParams(2)
        # self.autoFocus()
       
        self.imageRoi = {}
        self.imageRoi['roi']     = roi
        self.imageRoi['iterNum'] = self.iterNum

        for cond, exposure in self.imageConds.items():
            self.imageRoi[cond] = {}    
            self.imageRoi[cond]['imageStack'] = np.empty((self.cameraImage.shape[0], self.cameraImage.shape[1], numTiles), dtype = 'uint16')
            self.imageRoi[cond]['xyzTime']    = np.empty((numTiles, 4), dtype = 'float')
        
        for pos in range(numTiles):
            
            self.moveXYStage(scanPos[pos,0],scanPos[pos,1])
            self.mmc.waitForDevice(self.mP['xyDrive'])
            self.recordImage()  # Record Image will determine number of images to record this will also create an imageStack that will need to be copied.
            for cond, exposure in self.imageConds.items():
                np.copyto(self.imageRoi[cond]['imageStack'][:,:,pos], self.imageStack[cond]['image'])
                np.copyto(self.imageRoi[cond]['xyzTime'][pos, :],     self.imageStack[cond]['xyzTime'])

        self.saveRoiStack()
    
        return None
    
    def saveRoiStack(self):
        '''
        Save region of interst image stack in an additional thread to 
        free up the system to move to the next region of interset. 

        This will need to do two things.  Save a local copy to a drive 
        and mirro that copy to RSS.

        File naming convention.  'roiLabel'_XXXX.hdf5 
        '''

        roiLabel = self.imageRoi['roi']
        iterNum  = self.imageRoi['iterNum']

         # SaveTile Stack in sepqarate Thread
        iterNum = self.iterNum
        imageRoi= self.imageRoi.copy()
        
        zState  = {}
        zState['zStagePos'] = self.zStagePos.copy()
        zState['zFocusPos'] = self.zFocusPos.copy()

        mP = self.mP.copy()

        primarySavePath = pathlib.Path(self.primarySaveDir).joinpath(f'{roiLabel}')
        primaryFilePath =  primarySavePath /  f'{roiLabel}_{iterNum:04d}.hdf5'

        backupSavePath = pathlib.Path(self.backupSaveDir).joinpath(f'{roiLabel}')
        backupFilePath =  backupSavePath /  f'{roiLabel}_{iterNum:04d}.hdf5'
    
        filePaths = {'primary path': primaryFilePath,
                     'backup path' : backupFilePath}

        # saveObj = RoiSaveThread(filePaths, imageRoi, mP, zState)
        # saveObj.start()
            
        return None
    
    def updateStageZPos(self):
        '''
        TODO: Update this section to convolve the neighborings positions
        '''

        iterNum = self.iterNum
        self.zStagePos[:, iterNum+1] = self.zFocusPos[:, iterNum]

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

        microPrimaryPath = pathlib.Path.home() / '.microscopePaths'
        with open(microPrimaryPath, 'r') as fileIn:
            expDirectory = json.load(fileIn)

        self.primarySaveDir = expDirectory['primaryMicroPath']

        primaryMicroPropPath = pathlib.Path(self.primarySaveDir) / '.odelayMicroscopeConfig'
        primaryZStatePath    = pathlib.Path(self.primarySaveDir) / '.odelayZstate'

        with open(primaryMicroPropPath, 'r') as fileIn:
            self.mP = json.load(fileIn)

        zState = fio.loadData(primaryZStatePath)

        self.roiList = self.mP['roiList']
        self.iterNum = self.mP['iterNum']
        self.roi     = self.mP['roi']
        self.zStagePos = zState['zStagePos']
        self.zFocusPos = zState['zFocusPos']
        self.primarySaveDir = self.mP['primarySaveDir']
        self.backupSaveDir  = self.mP['backupSaveDir']
        # self.microscopeConfigFilePath = None`

    def initializeExperiment(self):
        '''
        1.Check for directories.
        2.Check for microscopeMp Files
        3.Check for iteration number
        '''

        self.iterNum = 0
        roiList = self.mP['roiList']

        if not self.primarySaveDir.exists():
            self.primarySaveDir.mkdir()

        if not self.backupSaveDir.exists():
            self.backupSaveDir.mkdir()

        for roi in roiList:
            primaryRoiPath = pathlib.Path(self.primarySaveDir).joinpath(f'{roi}')
            backupRoiPath  = pathlib.Path(self.backupSaveDir).joinpath(f'{roi}')
            
            if not primaryRoiPath.exists():
                primaryRoiPath.mkdir()

            if not backupRoiPath.exists():
                backupRoiPath.mkdir()

       
        self.zStagePos = np.zeros((len(roiList), self.mP['numTimePoints']+1), dtype = 'float')
        self.zStagePos[:,self.iterNum] = self.mP['XYZOrigin'][2] 

        self.zFocusPos = np.zeros((len(roiList), self.mP['numTimePoints']), dtype = 'float')
        self.mP['experimentInitialized'] = True



        return None

    def closeEvent(self, event):

        try:
            self.mmc.unloadAllDevices()
            self.mmc.reset()
            print('All devices unloaded and shut down')
        except:
            print('No devices loaded')

    def generateFocusCurve(self):


        zPos = self.mmc.getPosition(self.mP['zDrive'])
        zRange    = np.arange(-30,30,step = 0.5, dtype = 'float') + zPos
        numSteps  = zRange.shape[0]
        
        # Calculate Z-Movements
        zRecord = np.zeros((numSteps,2), dtype = 'float')
        zFocus = np.zeros((numSteps,2), dtype = 'float')    
        zFocus[:,0] = zRange
        
        for step in range(numSteps):
            zFocusPos = zFocus[step,0]
            self.mmc.setPosition(self.mP['zDrive'], zFocusPos)
            self.mmc.waitForDevice(self.mP['zDrive'])
            self.snapImage()
       
            zFocus[step,0] = self.mmc.getPosition(self.mP['zDrive'])
            zFocus[step,1] = cv2.Laplacian(self.cameraImage, cv2.CV_64F ).var()
            zRecord[step,:2] = zFocus[step,:2]
 

        focusData = {'xData': zFocus[:,0], 'yData': zFocus[:,1]}

        self.focusPlot = FocusPlot(focusData)
        self.focusPlot.show()
        return None
        
if __name__ == '__main__':

    originDir = pathlib.Path.cwd()
    odelayPath = pathlib.Path(__file__).parent.parent
    os.chdir(odelayPath)

    app = QApplication(sys.argv)
    mastercontrol = MasterControl()
    sys.exit(app.exec_())
    