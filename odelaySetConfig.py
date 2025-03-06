import json
import os
import pathlib
import getpass

''' 
This file sets up inital parameters for submitting jobs to cybertron
Please check this file regularly and update it with your personal 
information as needed.

Note the json file this generates can have additional parametes saved to it to better manipulate and customize the ODELAY-ODELAM data pipeline.
'''
def setConfig():
    PARENT_DIR =  f'/home/{getpass.getuser()}/'

    # for the PBSPro files.  These values are important to passing the directories where experimental data is written and read.
    environmentDict = { 
                'export IMGDIR':  None,
                'export DATDIR':  None}

    # Generally we won't need more than this but we could add more space if needed.
    resourcesDict = {
                'mem': '8gb',
                'ncpus': '1'}  

    odelayConfig = {
        'HOST':'cybertron',
        'K_PATH': os.path.expanduser('~\.ssh\id_rsa'),
        'JOB_TEMPLATE': 'pbs.sh',
        'PARENT_DIR': f'/home/{getpass.getuser()}/',
        'sponsor':'jaitch',
        'email':'Thurston.Herricks@SeattleChildrens.org',
        'name': 'testrun',
        'resources':   resourcesDict ,
        'environment': environmentDict,
        'queue': 'workq',
        'stdout': f'{PARENT_DIR}odelay_logs/stdout',
        'stderr': f'{PARENT_DIR}odelay_logs/stderr',
        'emails': 'abe',
        'cmd': 'sleep 5',
        'setup': '',
        'TemplateLocation': './templates',
        'LocalImageDir': None,
        'LocalDataDir':  None,
        'HPCImageDir':   None,
        'HPCDataDir':    None,
        'LocalDirPrefix':None,
        'HPCDirPrefix':  None, 
        'PathCheck':     False,
        'pbskeylist': ['sponsor', 'email', 'name', 'resources', 'environment', 'queue','stdout','stderr','emails','cmd','setup']
    }

    configfilePath = pathlib.Path( pathlib.Path.home() / '.odelayconfig' )

    with open(configfilePath, 'w') as configFile:
        json.dump(odelayConfig, configFile)

    return None