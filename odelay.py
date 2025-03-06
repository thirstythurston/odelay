import os
import getpass
import pathlib
import time
import subprocess
import json
import asyncio
import functools
from datetime import datetime

import click
import paramiko
import jinja2
import numpy as np
import tools.fileio     as fio
import tools.imagepl    as opl
import tools.odelayplot as odp
import odelaySetConfig

'''
ODELAY command line interface
This interface provides a few methods for processing and interacting with
ODELAY-ODELAM data.  

Basic commands:
odelay set-image-dir:  'STRING' this creates a environment variable IMGDIR to the ODELAY 
             directory hat should have the following files
             -folders named after each region of interest that is imaged
             -ODELAY_StageData.mat
             -ODELAY_Monitor.mat

odelay set-data-dir:   'STRING' this creates an environment variable EXPDIR where 
              a folder named ODELAY Roi Data will be created and a file
              XXX_Index_ODELAYData.hdf5 file will be created. This hdf5 file
              contains experiment variables that describe the microscope
              and will have local pointers to data files in the ODELAY Roi Data 
              directory

odelay initialize: If the image directory and experiment directory are set then the files
            XXX_Index_ODELAYData.hdf5 will be created and the ODELAY Roi Data directory

odelay process all:    This command will process an individual well and save the data to the 
            experiment directory / ODELAY Roi Data/.  

'''

#  Decorator magic
def background(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs)
    
    return wrapped

#################
# File Location #
#################

class Fileloc(object):

    def __init__(self, imgdirfile = None, expdirfile = None, configfile = None ):
        self.homedir   =  pathlib.Path.home()
        self.configfile = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

# ##############
# # PBS Client #
# ##############
class PBSClient:

    def __init__(self, host=None, pkey=None, waitforexit=True):
        self.host = host
        self.pkey = pkey
        self.jobs = {}
        self.waitforexit = waitforexit

    def __enter__(self):
        # setup ssh connection
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(hostname=self.host, username=os.getlogin(), pkey=self.pkey)

        # setup sftp connection
        self._sftp = self._ssh.open_sftp()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.waitforexit:
            grep_field = 'Exit_status'
        else:
            grep_field = 'job_state'
        
        while None in self.jobs.values():
            for job_id in self.jobs.keys():
                # check return codes
                if self.jobs[job_id]==None:
                    stdout = self._ssh.exec_command(f'qstat -xf {job_id} | grep {grep_field}')[1]
                    exit_status = stdout.read().decode('utf-8').strip() or None

                    # update returncode
                    self.jobs[job_id] = exit_status

            time.sleep(0.25)
        
        print('##############################################')
        for keys, values in self.jobs.items():
            print(keys, ':', values)
        print('##############################################')

        # close ssh and sftp connections
        self._sftp.close()
        self._ssh.close()

    def run_pbs(self, filepath, pbs, waitforexit):
        """Creates pbs on remote resource and executes it with
        qsub.

        :param filepath: full filepath for pbs on remote host
        :param content: pbs contents as string
        :return:
        """
        self.waitforexit = waitforexit 
        with self._sftp.file(filepath, 'w') as pbs_script:
            pbs_script.write(pbs)

            stdin, stdout, stderr = self._ssh.exec_command(f'qsub "{filepath}"', timeout=18000)
            # get pbs job id
            job_id = stdout.read().decode('utf-8').strip()

            # add job to heap
            self.jobs[job_id] = None

        print(f'Job {job_id} Submitted')

pass_econfig   = click.make_pass_decorator(Fileloc,   ensure= True)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'],
                        max_content_width=80)

@click.group(context_settings=CONTEXT_SETTINGS)
@pass_econfig
def cli(fileloc):
    
    if not pathlib.Path.exists(fileloc.configfile):
    
        odelaySetConfig.setConfig()

        click.echo('Create Config File at %s' % str(fileloc.homedir))

    return None

@cli.command()
@click.option('--loc-data-dir', prompt='Set the data directory', default = None)
@pass_econfig
def set_data_dir(fileloc, loc_data_dir):
    '''Set the directory where processed ODELAY data is loaded/saved'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localDataPath = pathlib.Path(loc_data_dir)

    if localDataPath.exists:
        resolvedPath = localDataPath.resolve()
        LocalDataPathstr  = str(resolvedPath)

        HPCDataPath  = LocalDataPathstr.replace('\\','/').replace('//helens','/')

        odelayConfig['LocalDataDir'] = loc_data_dir
        odelayConfig['HPCDataDir']   = HPCDataPath    
        click.echo(f'Data Directory path from local computer is: {loc_data_dir}')
        click.echo(f'Data Directory path from  HPC  computer is: {HPCDataPath}')
    

    odelayConfig['PathCheck'] = False

    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return None

@cli.command()
@click.option('--loc-image-dir', prompt='Set the image directory',  default = None)
@pass_econfig
def set_image_dir(fileloc, loc_image_dir):
    '''Set the directory where processed ODELAY data is loaded/saved'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localImagePath = pathlib.Path(loc_image_dir)

    if localImagePath.exists():
        resolvedPath = localImagePath.resolve()
        LocalImagePathstr  = str(resolvedPath)

        HPCImagePath  = LocalImagePathstr.replace('\\','/').replace('//pplhpc1ces','/gpfs/scratch')
        
        odelayConfig['LocalImageDir'] = loc_image_dir
        odelayConfig['HPCImageDir']   = HPCImagePath    
        click.echo(f'Image Directory path from local computer is: {loc_image_dir}')
        click.echo(f'Image Directory path from  HPC  computer is: {HPCImagePath}')
    

    odelayConfig['PathCheck'] = False

    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return None

@cli.command()
@click.option('--loc-image-dir', prompt='Set the image directory',  default = None)
@pass_econfig
def set_local_image_dir(fileloc, loc_image_dir):
    '''Set the directory where the experiment's images are located'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localImagePath = pathlib.Path(loc_image_dir)
  
    
    if localImagePath.exists():
        
        LocalImagePath  = pathlib.Path(loc_image_dir).resolve()
        LocalImagePathstr  = str(LocalImagePath)
        
        HPCImagePath = LocalImagePathstr.replace('\\','/').replace('//pplhpc1ces','/gpfs/scratch')

        odelayConfig['LocalImageDir'] = loc_image_dir
        click.echo(f'Image Directory path from local computer is: {LocalImagePathstr}')
    else:
        print('For some reason the local image path does not exist')

    odelayConfig['PathCheck'] = False
    
    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)


    return None

@cli.command()
@click.option('--loc-data-dir', prompt='Set the data directory', default = None)
@pass_econfig
def set_local_data_dir(fileloc, loc_data_dir):
    '''Set the directory where processed ODELAY data is loaded/saved'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localDataPath = pathlib.Path(loc_data_dir)

    breakpoint()

    if localDataPath.exists():
        resolvedPath = localDataPath.resolve()
        LocalDataPathstr  = str(resolvedPath)

        HPCDataPath  = LocalDataPathstr.replace('\\','/').replace('//helens','/')

        odelayConfig['LocalDataDir'] = LocalDataPathstr
       
        click.echo(f'Data Directory path from local computer is: {odelayConfig["LocalDataDir"]}')
       
    

    odelayConfig['PathCheck'] = False

    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return None

@cli.command()
@click.option('--loc-image-dir', prompt='Set the image directory',  default = None)
@pass_econfig
def set_hpc_image_dir(fileloc, loc_image_dir):
    '''Set the directory where the experiment's images are located'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localImagePath = pathlib.Path(loc_image_dir)
    
    if localImagePath.exists():
        
        LocalImagePath  = pathlib.Path(loc_image_dir).resolve()
        LocalImagePathstr  = str(LocalImagePath)
        
        HPCImagePath = LocalImagePathstr.replace('\\','/').replace('//pplhpc1ces','/gpfs/scratch').replace('//baker','/').replace('//Archive', '//archive')

        odelayConfig['HPCImageDir'] =  HPCImagePath
        click.echo(f'Image Directory path from local computer is: { HPCImagePath}')

    odelayConfig['PathCheck'] = False
    
    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)


    return None

@cli.command()
@click.option('--loc-data-dir', prompt='Set the data directory', default = None)
@pass_econfig
def set_hpc_data_dir(fileloc, loc_data_dir):
    '''Set the directory where processed ODELAY data is loaded/saved'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    localDataPath = pathlib.Path(loc_data_dir)

    if localDataPath.exists:
        resolvedPath = localDataPath.resolve()
        LocalDataPathstr  = str(resolvedPath)

        HPCDataPath  = LocalDataPathstr.replace('\\','/').replace('//helens','/').replace('//baker','/').replace('Archive', 'archive')

        odelayConfig['HPCDataDir'] = HPCDataPath
       
        click.echo(f'Data Directory path from local computer is: {HPCDataPath}')

    odelayConfig['PathCheck'] = False

    with open(fileloc.configfile, 'w') as fileOut:
        json.dump(odelayConfig, fileOut)

    return None

@cli.command()
@pass_econfig
def current_experiment(fileloc):
    '''List the current experiment directories'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    loc_data_dir = odelayConfig['LocalDataDir']     
    click.echo(f'Data Directory path from local computer is: {loc_data_dir}')

    hpc_data_dir = odelayConfig['HPCDataDir']     
    click.echo(f'Data Directory path from HPC computer is: {hpc_data_dir}')

    loc_data_dir = odelayConfig['LocalImageDir']     
    click.echo(f'Image Directory path from local computer is: {loc_data_dir}')

    hpc_data_dir = odelayConfig['HPCImageDir']     
    click.echo(f'Image Directory path from HPC computer is: {hpc_data_dir}')

    return None

@cli.command()
@click.argument('roi', type=str)
@pass_econfig
def process(fileloc, roi):
    '''Process region of interest or whole experiment'''
   
    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])

    indexList = [*dataPath.glob('*Index_ODELAYData.*')]

    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
        
        expData = fio.loadData(expIndexPath)
        roiList = [*expData['roiFiles']]
        roiList.sort()
      
        if roi == 'all':
            processRoiAll(odelayConfig, roiList)

        elif roi in roiList:
            # click.echo('This should not block')
            processRoiAll(odelayConfig, [roi])
            # click.echo('This is after the function that should not block')
    else:
        click.echo('Could not find the correct index file or there were more than one in the diretory')


    return None


@cli.command()
@click.argument('roi', type=str)
@pass_econfig
def process_local(fileloc, roi):
    '''Process region of interest or whole experiment'''
   
    if roi == 'all':
        roiData = opl.odelay_localProcess(roiList = None, process_count = None)
    else:
        roiData = opl.odelay_localProcess(roiList = [roi], process_count = 1)


    return None


@cli.command()
@click.argument('roi', type=str)
@pass_econfig
def process_mac(fileloc, roi):
    '''Process region of interest or whole experiment'''
   
    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])

    indexList = [*dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
        
        expData = fio.loadData(expIndexPath)
        roiList = [*expData['roiFiles']]
        roiList.sort()
      
        if roi == 'all':
            processMacAll(odelayConfig, roiList)

        elif roi in roiList:
            # click.echo('This should not block')
            
            processMacAll(odelayConfig, [roi])
            # click.echo('This is after the function that should not block')
    else:
        click.echo('Could not find the correct index file or there were more than one in the diretory')


    return None

@cli.command()
@pass_econfig
def initialize(fileloc):
    '''Initialize experiment file setup by sending a PBS file to the cluster and checking that file paths work'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)
    click.echo(odelayConfig['LocalImageDir'])

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    isImgPath = pathlib.Path.exists(imagePath)
    isDatPath = pathlib.Path.exists(dataPath)

    # if both the data and image paths are real then write 
    # the odelayConfig.json file to the data directory.    
    if isImgPath and isDatPath:
        dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
        configDataPath = dataPath / 'odelayConfig.json'
        with open(configDataPath, 'w') as fileOut:
            json.dump(odelayConfig, fileOut)
        click.echo("Local paths are correct")
        click.echo("Asking hpc to check its data and image paths")

        # Generate PBS dictionary 
        pbsDict ={}
        for key in odelayConfig['pbskeylist']:
            pbsDict[key] = odelayConfig[key]
        pbsDict['name'] = imagePath.name.replace(" ","_") +'_init'
        pbsDict['cmd'] = 'odelay hpc-init' 
        imgdir = odelayConfig['HPCImageDir']
        datdir = odelayConfig['HPCDataDir']
        pbsDict['environment']['export IMGDIR'] = f'"{imgdir}"'
        pbsDict['environment']['export DATDIR'] = f'"{datdir}"'

        # get id_rsa file for private key
        passkey = paramiko.RSAKey.from_private_key_file(odelayConfig['K_PATH'])
        # id the template folder
        templates_path = pathlib.Path(__file__).parent / "templates"
        fs_loader = jinja2.FileSystemLoader(searchpath= str(templates_path))
        # create jinja2 environment for creating a form
        env = jinja2.Environment(loader=fs_loader)
        # load in the form that variables will be added to
        
        template = env.get_template(odelayConfig['JOB_TEMPLATE']) 
        # id the file name and path to the file.
        # TODO:  alter this for odelay path
        filepath = f'{odelayConfig["PARENT_DIR"]}odelay_logs/{imagePath.name.replace(" ","_")}-init-odelay.sh' 

        pbsRen = template.render(pbsDict)

        click.echo('The File Path is: %s' % filepath)
     
        with PBSClient(host = odelayConfig['HOST'], pkey=passkey) as cli:
            cli.run_pbs(filepath = filepath, pbs=pbsRen, waitforexit = True)

        with open(configDataPath,'r') as fileIn:
            checkConfig = json.load(fileIn)

        if checkConfig['PathCheck']:
            click.echo('Experiment is Initialized')

        else:
            click.echo('Experiment is not Initized.  A directory was not found')


    else:
        if isImgPath is False:
            click.echo('The Image directory is %s does not exist' % odelayConfig['LocalImageDir'])
            click.echo('Please use command: odelay set-image-dir -loc to set the correct image directory')

        if isDatPath is False:
            click.echo('The Data directory is %s does not exist' % odelayConfig['LocalDataDir'])
            click.echo('Please use command: odelay set-data-dir -loc to set the correct image directory')

    # click.echo('The directory is %i!' % expconfig.iterator) 
    return None

@cli.command()
@click.argument('roiid', nargs=2, type=click.Tuple([str, str]))
@pass_econfig
def plot_gc(fileloc, roiid):
    '''
    Plot a growth curve figure with the Region of Intrest
    use:  plot_gc roi organism
    roi is single region of interest. Usually E07 or a well ID
    organism is currently Yeast, Mtb, and Mabs
    '''
    
    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)
    
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    roi = roiid[0]
    organism = roiid[1]

    if roi == 'all':
        summaryList = list(dataPath.glob('*summary.hdf5'))
        roiFolder   = list(dataPath.glob('ODELAY Roi Data'))
        
        if len(summaryList)==1 and summaryList[0].exists():
            expDict = fio.loadData(summaryList[0])
            click.echo('summary loaded')

            saveFolder = dataPath / 'ODELAY Growth Curves'
            if not saveFolder.exists():
                saveFolder.mkdir()

            for roi in expDict.keys():
                try:
                    savePath = saveFolder / f'{roi}.png'
                    odp.figPlotGCs(expDict[roi], organism = organism, saveAll=True, savePath = savePath )
                except:  
                    click.echo(f"{roi} didn't print")
        else:
            click.echo('Summary file does not exist please make one')

    else:
        
        summaryList = list(dataPath.glob('*summary.hdf5'))
        roiFolder   = list(dataPath.glob('ODELAY Roi Data'))
        # roiPath  = roiFolder[0] / f'{roi}.hdf5'

        if len(summaryList)==1 and summaryList[0].exists():
            expDict = fio.loadData(summaryList[0])
            click.echo('summary loaded')
            if roi in expDict.keys():
                
                odp.figPlotGCs(expDict[roi], organism = organism)

        elif len(roiFolder)==1 and roiFolder[0].joinPath(f'{roi}.hdf5').exists():
            roiPath = roiFolder[0].joinPath(f'{roi}.hdf5')
            roiData = fio.loadData(roiPath)
            rc = roiData['fitData'].shape
            idVec = np.arange(rc[0], dtype = 'uint32')
            inds = roiData['fitData'][:,0]>0
            
            roiDict = {}
            roiDict['fitData']    = roiData['fitData'][inds,:]
            roiDict['objectArea'] = roiData['objectArea'][inds,:]
            roiDict['timePoints'] = roiData['timePoints']
            roiDict['objID']      = idVec[inds]
            roiDict['roi']        = roi
            roiDict['roiInfo']    = roiData['roiInfo']

            odp.figPlotGCs(roiDict, organism = organism)

        else:
            click.echo('Path to Roi file or to summary file broken.')
            click.echo('Please check local data path to experiment folder')
        

    
    return None    
    
@cli.command()
@click.argument('organism', nargs=1, type=click.STRING, default=None)
@pass_econfig
def plot_summary(fileloc, organism):
    '''plot summary figure of entire experiment'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)
    
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    
    summaryList = list(dataPath.glob('*summary.hdf5'))
    
    if len(summaryList)==1 and summaryList[0].exists():
        summaryFilePath = summaryList[0]
        expDict = fio.loadData(summaryFilePath)
        click.echo('summary loaded')
        title = summaryFilePath.name.replace('.hdf5', '')
        odp.figExpSummary(expDict, organism, title)


    return None

@cli.command()
@pass_econfig
def listroi(fileloc):
    '''List the experiment ROI and the experiment parameters'''
    click.echo('This will list all of the roi present in the current experimet')

@cli.command()
@click.argument('roiind', type=str)
@pass_econfig
def export_csv(fileloc, roiind):
    ''' Export CSV file tables of Growth Curves and fit data. '''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    
    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
        expData = fio.loadData(expIndexPath)
        expKeys = [*expData]

        if 'roiFiles' in expKeys:
            roiList = [*expData['roiFiles']]
            roiList.sort()

        elif 'ImageVars' in expKeys:
            roiList = [*expData['ImageVars']['StrainID']]
            roiList.sort()
        
        
        if roiind == 'all':
            fio.exportcsv(dataPath, roiList)
        elif roiind in roiList:
            fio.exportcsv(dataPath, [roiind])

    else:
        click.echo('Roi ID was not in the experiment Roi List')
    
    return None

@cli.command()
@click.argument('roi', required=True, type=str)
@pass_econfig
def export_avi(fileloc, roi):
    ''' Export CSV file tables of Growth Curves and fit data. '''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    indexList = [k for k in dataPath.glob('*Index_ODELAYData.*')]
    if len(indexList)==1:
        expIndexPath = dataPath / indexList[0]
        
        expData = fio.loadData(expIndexPath)
        roiList = [*expData['roiFiles']]
        roiList.sort()
        
        if roi == 'all':
            renderAvi(odelayConfig, roiList)

        elif roi in roiList:
            # click.echo('This should not block')
            renderAvi(odelayConfig, [roi])
    
    else:
        click.echo('Roi ID was not in the experiment Roi List')
    # df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv')

    return None

@cli.command()
@click.argument('roi', type=str)
@pass_econfig
def export_tiffs(fileloc, roi):
    ''' Export CSV file tables of Growth Curves and fit data. '''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    expIndexPath = [*dataPath.glob('*Index_ODELAYData.*')]
    if len(expIndexPath)==1:
        expData = fio.loadData(expIndexPath[0])
        roiList = [*expData['roiFiles']]
        roiList.sort()

        if roi == 'all':
            renderTiffs(odelayConfig, roiList)

        elif roi in roiList:
            # click.echo('This should not block')
            renderTiffs(odelayConfig, [roi])

    else:
        click.echo('Roi ID was not in the experiment Roi List')
    # df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv')

    return None

@cli.command()
@pass_econfig
def summarize_experiment(fileloc):
    ''' Sumerize Experiment into dictionary that only contains Fit Data and Object Area'''

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)

    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    if dataPath.joinpath('ODELAY Well Data').exists():
        expDict = fio.summarizeMatLabExp(dataPath, True)

    elif dataPath.joinpath('ODELAY Roi Data').exists():
        expDict = fio.summarizeExp(dataPath, True)
    
    else:
        click.echo('Data files not found.')

    return None

@cli.command()
@pass_econfig
@click.argument('imageid', nargs=2, type=click.Tuple([str, int]))
def showroi(fileloc, imageid):
    ''' Display Region Of Interest in viewer'''
    click.echo('roi ID %s image number = %d' % imageid)
    roiID    = imageid[0]
    imageNum = imageid[1]

    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)
    
    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])

    click.echo('Image Directory: %s' % str(imagePath))
    click.echo('Data/Save Directory: %s' % str(dataPath))
    
    click.echo('Image Number %i' % imageNum)
    k = odp.showImage(roiID, imageNum, imagePath, dataPath)

    # create loop where the keys '[' ']' advances and decrements images
    while k != -1:
        if   k == 93: 
            imageNum += 1
            click.echo('Image Number %i' % imageNum)
            k = odp.showImage(roiID, imageNum, imagePath, dataPath)
        elif k == 91:
            imageNum -= 1
            click.echo('Image Number %i' % imageNum)
            k = odp.showImage(roiID, imageNum, imagePath, dataPath)
        else:
            click.echo('Image Number %i' % imageNum)
            k = odp.showImage(roiID, imageNum, imagePath, dataPath)

    return None

@cli.command()
@pass_econfig
@click.argument('imageid', nargs=2, type=click.Tuple([str, int]))
def stitchimage(fileloc, imageid):
    '''View an image from a specific ROI and image number eg: odelay stitch image E06 19'''
    click.echo('roi ID %s image number = %d' % imageid)
    roiID    = imageid[0]
    imageNum = imageid[1]


    with open(fileloc.configfile, 'r') as fileIn:
        odelayConfig = json.load(fileIn)
    
    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])

    
    key = odp.stitchIm(roiID, imageNum,  imagePath, dataPath)

    # create loop where the keys '[' ']' advances and decrements images
    while key != -1:
        if   key == 93: 
            imageNum += 1
            click.echo('Image Number %i' % imageNum)
            key = odp.stitchIm(roiID, imageNum, imagePath, dataPath)
        elif key == 91:
            imageNum -= 1
            click.echo('Image Number %i' % imageNum)
            key = odp.stitchIm(roiID, imageNum, imagePath, dataPath)
        else:
            click.echo('Image Number %i' % imageNum)
            key = odp.stitchIm(roiID, imageNum, imagePath, dataPath)

    return None

@cli.command()
def image_viewer():
    odp.imageViewer()

@cli.command()
def video_viewer():

    odp.videoViewer()
    
    return None

#################################################################
# HPC callable functions:  The following functions are designed 
# to be called from a remote host.  They have the following 
# assumptions.  IMGDIR and DATDIR are environment variables 
# that correctly point to directory paths.  All functions that 
# accept roi can be asyncronously as their output is not 
# dependant on follow up tasks.
################################################################

@cli.command()
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_init(imgdir, datdir):
    '''Initialize experiment file setup by sending a PBS file to the cluster and checking that file pats work'''
    # Check for odelay config file that should have been written with the Initialize function
    click.echo(imgdir)
    imagePath = pathlib.Path(imgdir)
    dataPath  = pathlib.Path(datdir)
    if pathlib.Path.exists(dataPath) and pathlib.Path.exists(imagePath):
        # The odelayConfig.json file should be there and if it is open
        # and set PathCheck to true
        configPath = dataPath / 'odelayConfig.json'
        with open(configPath, 'r') as fileIn:
            odelayConfig = json.load(fileIn)

        # Now set up experiment.  
        expDictionary = opl.initializeExperiment(imagePath, dataPath)

        odelayConfig['PathCheck'] = True

        with open(configPath, 'w') as fileOut:
            json.dump(odelayConfig, fileOut)

    return None

@cli.command()
@click.option('-r','--roiind', required=True)
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_process(roiind, imgdir, datdir):
    '''Callable function from PBSPRO file HPC computing'''

    # Get image file and data data file dir
    imagePath = pathlib.Path(imgdir)
    dataPath  = pathlib.Path(datdir)
    roiData = opl.roiProcess(imagePath, dataPath, roiind)

    return None 

@cli.command()
@click.option('-r','--roiind', required=True)
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_mac(roiind, imgdir, datdir):
    '''Callable function from PBSPRO file HPC computing'''

    # Get image file and data data file dir
    imagePath = pathlib.Path(imgdir)
    dataPath  = pathlib.Path(datdir)
    roiData = opl.roiMacInfo(imagePath, dataPath, roiind)

    return None 

@cli.command()
@click.option('-r', '--roiind', required=True)
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_avi(roiind, imgdir, datdir):
    '''Callable function from PBSPRO file HPC computing to export avi file tables of Growth Curves and fit data. '''

    dataPath = pathlib.Path(datdir)
    imagePath = pathlib.Path(imgdir)
    fio.exportavi(imagePath, dataPath, roiind)

    return None

@cli.command()
@click.option('-r', '--roiind', required=True)
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_tiff(roiind, imgdir, datdir):
    '''Callable function from PBSPRO file HPC computing to export tiff files.'''

    dataPath = pathlib.Path(datdir)
    imagePath = pathlib.Path(imgdir)
    fio.exporttiffs(imagePath, dataPath, roiind)

    return None

@cli.command()
@click.option('--imgdir', envvar ='IMGDIR')
@click.option('--datdir', envvar ='DATDIR')
def hpc_checkpaths(imgdir, datdir):
    ''' Callable function from PBSPRO file to check that paths to Directories are correct'''

    imagePath = pathlib.Path(imgdir)
    dataPath  = pathlib.Path(datdir)
    configPath = dataPath / 'configFile.json'

    isimPath = pathlib.Path.exists(imagePath)
    isconfigPath = pathlib.Path.exists(configPath)

    if isconfigPath:    
        with open(configPath, 'r') as fileIn:
            odelayConfig = json.load(fileIn)

        if isimPath:
            odelayConfig['PathCheck'] = [True]
    
    return None 

#######################################
# PBSClient Generators
#######################################
def processRoiAll(odelayConfig, roiList):

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    isImgPath = pathlib.Path.exists(imagePath)
    isDatPath = pathlib.Path.exists(dataPath)
    
    if isImgPath and isDatPath:
    
        # Generate PBS dictionary 
        pbsDict ={}
        for key in odelayConfig['pbskeylist']:
            pbsDict[key] = odelayConfig[key]
        
        imgdir = odelayConfig['HPCImageDir']
        datdir = odelayConfig['HPCDataDir']
        # export the Image directory as a variable command for the PBS file
        # the quotation marks are important here.
        pbsDict['environment']['export IMGDIR'] = f'"{imgdir}"' 
        pbsDict['environment']['export DATDIR'] = f'"{datdir}"'
        # get id_rsa file for private key
        passkey = paramiko.RSAKey.from_private_key_file(odelayConfig['K_PATH'])
        # id the template folder
        templates_path = pathlib.Path(__file__).parent / "templates"
        fs_loader = jinja2.FileSystemLoader(searchpath= str(templates_path))
        # create jinja2 environment for creating a form
        env = jinja2.Environment(loader=fs_loader)
        # load in the form that variables will be added to
        template = env.get_template(odelayConfig['JOB_TEMPLATE']) 
        # id the file name and path to the file.
        with PBSClient(host = odelayConfig['HOST'], pkey=passkey) as cli:
            for roi in roiList:
                # Set the command to run on the HPC
                pbsDict['cmd'] = f'odelay hpc-process -r {roi}'
                 # Name the file, generate the file path for the bash script, and render the file
                pbsName = imagePath.name.replace(" ","_")
                pbsDict['name'] = f'{pbsName}-{roi}'
                filepath = f'{odelayConfig["PARENT_DIR"]}odelay_logs/{pbsName}-{roi}-odelay.sh'
                pbsRen = template.render(pbsDict)
                #  use the PBSClient to run the pbs script remotely.  
                cli.run_pbs(filepath = filepath, pbs=pbsRen, waitforexit = False)
                # time.sleep(1)
                

    return None

def processMacAll(odelayConfig, roiList):

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath = pathlib.Path(odelayConfig['LocalDataDir'])
    isImgPath = pathlib.Path.exists(imagePath)
    isDatPath = pathlib.Path.exists(dataPath)
     
    if isImgPath and isDatPath:
    
        # Generate PBS dictionary 
        pbsDict ={}
        for key in odelayConfig['pbskeylist']:
            pbsDict[key] = odelayConfig[key]
        
        imgdir = odelayConfig['HPCImageDir']
        datdir = odelayConfig['HPCDataDir']
        # export the Image directory as a variable command for the PBS file
        # the quotation marks are important here.
        pbsDict['environment']['export IMGDIR'] = f'"{imgdir}"' 
        pbsDict['environment']['export DATDIR'] = f'"{datdir}"'
        # get id_rsa file for private key
        passkey = paramiko.RSAKey.from_private_key_file(odelayConfig['K_PATH'])
        # id the template folder
        templates_path = pathlib.Path(__file__).parent / "templates"
        fs_loader = jinja2.FileSystemLoader(searchpath= str(templates_path))
        # create jinja2 environment for creating a form
        env = jinja2.Environment(loader=fs_loader)
        # load in the form that variables will be added to
        template = env.get_template(odelayConfig['JOB_TEMPLATE']) 
        # id the file name and path to the file.
        with PBSClient(host = odelayConfig['HOST'], pkey=passkey) as cli:
            for roi in roiList:
                pbsDict['cmd'] = f'odelay hpc-mac -r {roi}'
                pbsName = imagePath.name.replace(" ","_")
                pbsDict['name'] = f'{pbsName}-{roi}-omacs'
                filepath = f'{odelayConfig["PARENT_DIR"]}odelay_logs/{pbsName}-{roi}-omacs.sh'
                pbsRen = template.render(pbsDict)
                cli.run_pbs(filepath = filepath, pbs=pbsRen, waitforexit = False)
        
    else:
        click.echo('Either ImagePath or DataPath is incorrect')

    return None

def renderAvi(odelayConfig, roiList):

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    isImgPath = pathlib.Path.exists(imagePath)
    isDatPath = pathlib.Path.exists(dataPath)
    
    if isImgPath and isDatPath:
    
        # Generate PBS dictionary 
        pbsDict ={}
        for key in odelayConfig['pbskeylist']:
            pbsDict[key] = odelayConfig[key]
        
        imgdir = odelayConfig['HPCImageDir']
        datdir = odelayConfig['HPCDataDir']
        # export the Image directory as a variable command for the PBS file
        # the quotation marks are important here.
        pbsDict['environment']['export IMGDIR'] = f'"{imgdir}"' 
        pbsDict['environment']['export DATDIR'] = f'"{datdir}"'
        # get id_rsa file for private key
        passkey = paramiko.RSAKey.from_private_key_file(odelayConfig['K_PATH'])
        # id the template folder
        templates_path = pathlib.Path(__file__).parent / "templates"
        fs_loader = jinja2.FileSystemLoader(searchpath= str(templates_path))
        # create jinja2 environment for creating a form
        env = jinja2.Environment(loader=fs_loader)
        # load in the form that variables will be added to
        template = env.get_template(odelayConfig['JOB_TEMPLATE']) 
        # id the file name and path to the file.
        with PBSClient(host = odelayConfig['HOST'], pkey=passkey) as cli:
            for roi in roiList:
                pbsDict['cmd'] = f'odelay hpc-avi -r {roi}'
                pbsName = imagePath.name.replace(" ","_")
                pbsDict['name'] = f'{pbsName}-{roi}-avi'
                filepath = f'{odelayConfig["PARENT_DIR"]}odelay_logs/{pbsName}-{roi}-avi.sh'
                pbsRen = template.render(pbsDict)
                cli.run_pbs(filepath = filepath, pbs=pbsRen, waitforexit = False)

    return None

def renderTiffs(odelayConfig, roiList):

    imagePath = pathlib.Path(odelayConfig['LocalImageDir'])
    dataPath  = pathlib.Path(odelayConfig['LocalDataDir'])
    isImgPath = pathlib.Path.exists(imagePath)
    isDatPath = pathlib.Path.exists(dataPath)
    
    if isImgPath and isDatPath:
    
        # Generate PBS dictionary 
        pbsDict ={}
        for key in odelayConfig['pbskeylist']:
            pbsDict[key] = odelayConfig[key]
        
        imgdir = odelayConfig['HPCImageDir']
        datdir = odelayConfig['HPCDataDir']
        # export the Image directory as a variable command for the PBS file
        # the quotation marks are important here.
        pbsDict['environment']['export IMGDIR'] = f'"{imgdir}"' 
        pbsDict['environment']['export DATDIR'] = f'"{datdir}"'
        # get id_rsa file for private key
        passkey = paramiko.RSAKey.from_private_key_file(odelayConfig['K_PATH'])
        # id the template folder
        templates_path = pathlib.Path(__file__).parent / "templates"
        fs_loader = jinja2.FileSystemLoader(searchpath= str(templates_path))
        # create jinja2 environment for creating a form
        env = jinja2.Environment(loader=fs_loader)
        # load in the form that variables will be added to
        template = env.get_template(odelayConfig['JOB_TEMPLATE']) 
        # id the file name and path to the file.
        with PBSClient(host = odelayConfig['HOST'], pkey=passkey) as cli:
            for roi in roiList:
                pbsDict['cmd'] = f'odelay hpc-tiff -r {roi}'
                pbsName = imagePath.name.replace(" ","_")
                pbsDict['name'] = f'{pbsName}-{roi}-tiff'
                filepath = f'{odelayConfig["PARENT_DIR"]}odelay_logs/{pbsName}-{roi}-tiff.sh'
                pbsRen = template.render(pbsDict)
                cli.run_pbs(filepath = filepath, pbs=pbsRen, waitforexit = False)

    return None