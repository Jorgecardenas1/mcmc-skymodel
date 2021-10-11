"""
Author: Jorge H. Cárdenas
University of Antioquia

"""
import os
import secrets


try:
    import jupyterthemes
except:
    os.system('pip install jupyterthemes')

    
    import jupyterthemes

try:
    import pylab
except:
    os.system('pip install pylab')
    
    import pylab
try:
    import tqdm
except:
    os.system('pip install tqdm')
    
    import tqdm
    
    
try:
    import arviz
except:
    os.system('pip install arviz')
    
    import arviz as az
    
    
try:
    import corner
except:
    os.system('pip install corner')
    
    import corner

try:
    import scipy
except:
    os.system('pip install scipy')
    import scipy

os.system('pip install jupyter_contrib_nbextensions')

from tqdm import tqdm
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import HTML

from bokeh.plotting import figure, show
from scipy import stats

from sklearn.neighbors import KernelDensity
from statsmodels.graphics.tsaplots import plot_acf  
