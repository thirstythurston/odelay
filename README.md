ODELAY!!!
=========
# Project Overview:

ODELAY tools is a python module implented in the publication: 

    One-Cell Doubling Evaluation by Living Arrays of Yeast, ODELAY!
    PMID: 27856698 PMCID: PMC5217116 DOI: 10.1534/g3.116.037044

This module has a interface for controlling inverted microscopes, automating the collection of images with a microscope, a pipeline for processing the images collected, and various plotting, summary and reporting tools for the data generated.
It was mainly designed for Leica DMI 6000 series microscopes and for Nikon Te2000 series instruments.  The microscopes are controlled using the MicroManager API and PyQt5.  


## Project Structure:
-----------------

    .
    ├── README.md 
    ├── odelay
    │   ├── __init__.py
    │   ├── experimentModules.py
    │   ├── fileio.py
    │   ├── imageviewer.py
    |   ├── imagepl.py
    │   ├── odealyMicroscopeControl.py
    |   └── odelayplot.py
    ├── odelaySetConfig.py
    ├── odelay.py
    └── pyproject.toml


# Installing:


First download and install anaconda for python 3.7 or greater.  While installing check to make sure that vscode is also installed with Anaconda
Next start up VScode and also find the *.exe for VScode C:\Users\therri\AppData\Local\Programs\Microsoft VS Code\Code.exe
Right click on the Code.exe and pin that to your home screen and possibly the task bar.  VScode is a really good editor and works really well with python.
Finally start vccode by double clicking on code
go File->Preferences->Extensions or Ctrl + Shift + X and go to extensions.  
Search for two extension packs, Anaconda Extension Pack and Python.
Install both extensions and anything else you think you might like.  
Now to start a new Python terminal go Crtl + Shift + P and select new python terminal.  This should set up a terminal with the (base) python environment

## create conda envitronment:

    conda create python=3.7 --name odelay

install odelay in editable mode by navigating to the directory where the odelay directory is installed
    
    pip install -e .

This will install OdelayTools.  

## Usefull settings for VSCode

Add the following to the vscode user settings.json file.  At $USER put the path from your user name under windows into that directory.  
These settings are good to set up python and start in the ODELAY environment automatically.    

    {
        "python.linting.pylintEnabled": true,
        "python.linting.enabled": true,
        "editor.showFoldingControls": "always",
        "python.pythonPath": "C:\\Users\\$USER\\AppData\\Local\\Continuum\\anaconda3\\envs\\odelay\\python.exe",
        "terminal.integrated.shell.windows": "C:\\WINDOWS\\System32\\cmd.exe",
        "files.eol": "\n",
        "window.zoomLevel": 0,
        "python.dataScience.sendSelectionToInteractiveWindow": false,
        "git.confirmSync": false,
        "python.linting.pylintArgs": ["--generate-members"]
        
    }


## Generate Keys for automated PBS script submission

The next step assumes that PBSpro is installed on the local cluseter to manage job submission.  These steps are not guaranteed to work.  For this step we need to generate id_rsa pass keys.  To do this first log into the cluster.  In this case the cluster is named cybertron.


    ssh cybertron

and log into cybertron. The first step is to set up keys such that you can log into cybertron without a password. 
    
    ssh-keygen -t rsa

The next step was to copy the id_rsa.pub and id_rsa files to the correct directories:  
    
    cat id_rsa.pub >> authorized_keys 
    chmod 400
    git config --global  --unset http.sshVerify

and then move it from the directory to save it in the local windows machine  

    mv id_rsa /active/(to the directory you have access to)  

Then move the file id_rsa to the .ssh directory under C:\Users\(~your ID)\.ssh
Also add the id_rsa.pub key into /.ssh/authorized_keys file on cybertron.  
This put the keys in the correct directories and allowed communication with cybertron without a password.

the next step is to install miniconda into your environment on cybertron.  
download miniconda to home directory on cybertron

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh

make sure the (base) shell is activated:

    conda create python=3.7 --name odelay
    conda activate odelay
    pip install -e . (or where ever the odelay package is installed if in a folder)

This will ensure that a python enviroment exists on cybertron and is the same as the one you had previously.
Now you are ready to start processing ODELAY files both locally and on cybertron.


## Conda Tips

List available conda enviromnets

    conda info --envs

Activate conda environment do sequentially:

    C:/Users/(username)/AppData/Local/Continuum/anaconda3/Scripts/activate
    conda activate base
    conda activate odelay (whatever you named the odelay environment I would suggest odelay or something similar)


# ODELAY Overview
---------------

ODELAY Commands:

    curexp                List the current experiment directories  
    export-avi            Export avi files use command :>odelay export-avi roi where roi is the ID of the spot or all for all videos   
    export-csv            Export CSV file tables eg :> odelay export-csv roi or all  
    hpc-avi               Callable function from PBSPRO file not for command line use  
    hpc-checkpaths        Callable function from PBSPRO file not for command line use  
    hpc-init              Callable function from PBSPRO file not for command line use  
    hpc-process           Callable function from PBSPRO file not for command line use  
    initialize            Initialize experiment files for processing   
    listroi               List the experiment ROI not currently working   
    plot-gc               Plot a growth curve figure with the Region of interest :>odelay plot-gc roi  
    process               Process region of interest or whole experiment on remote cluster  
    process_locals        Process region of interest on local computer  
    set-data-dir          Set the directory where processed ODELAY data is to be written  
    set-image-dir         Set the directory where the experiment's images are stored  
    showroi               Show how the image was stitched from a roi and timepoint :>odelay showroi roi image-number  
    stitchimage             
    summarize-experiment  Sumerize Experiment into dictionary that can be fed into plotting figure functions  

enter a command by typing odelay curexp.  This will list the experiment directory and the data directory. 
set the image and data directories to the experiment you wish to process.  The Image dir is where the images are stored.
the data directory is where the processed data files will be written.  


Instructions to process data:
Two directories need to be set for the program to read the data and write the processed data to.      
These are the: 
image directory where the odelay images are stored
data directory where precessed odelay data is written. 
    
To set the image directory:

    :> odelay set-image-dir 

and at the prompt paste the image directory location into the prompt eg E:\some dir\some other dir\data folder

To set the data directory:
    
    :> odelay set-data-dir 

and at the prompt paste the date directory location into the prompt eg G:\a dir\your bosses named directory\data folder
Initialize the data for processing on the HPC:
    
    :> odelay initialize

you will see the program check the directories that you entered to make sure they are correct.  
If it fails then there is an issue with resolving paths to th directories which will require debugging

use the command :

    >odelay process all
    
this will execute processing all the regions of interest on the HPC cluster

once the processing is done summarize the experiment
    
    :> odelay summarize-experiment

This will generate a single file with only the growth curves and the parameters fit to those growth curves.  
This will also assign labels for plotting the growth curves.

