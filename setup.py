# import setup tools
from setuptools import setup, find_packages

setup(
    name='OdelayTools',
    version ='0.1.0',
    author='Thurston Herricks', 
    description='ODELAY Tools and Image Pipeline',
    license='MIT',
    python_requires='>=3.7',
    py_modules=['odelay'],
    install_requires=[
        'Click',
        'fast_histogram',
        'ipython',
        'h5py',
        'jinja2',
        'jupyterlab',
        'matplotlib',
        'numpy',
        'pandas',
        'paramiko',
        'pylint',
        'PyQt5',
        'PyQtChart',
        'pyserial',
        'opencv-python',
        'opencv-contrib-python',
        'openpyxl',
        'sqlalchemy',
        'scipy',
        'xlrd',
    ],
    packages=find_packages(),
    entry_points='''
    [console_scripts]
    odelay=odelay:cli
    ''',
)
