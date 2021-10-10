"""
Author: Jorge H. Cárdenas
University of Antioquia
"""

import numpy as np
import math
from scipy.stats import lognorm

# Actual Beta PDF.
def beta(a, b, i):
    e1 = ss.gamma(a + b)
    e2 = ss.gamma(a)
    e3 = ss.gamma(b)
    e4 = i ** (a - 1)
    e5 = (1 - i) ** (b - 1)
    return (e1/(e2*e3)) * e4 * e5

def exponential_pdf(beta,x):
    return (1/beta)* np.exp(-x/beta)

def Gaussian_pdf(nu, sigma,x):
     #nu is the mean and sigma the standard deviation
    coeff = 1/(math.sqrt(2*math.pi*sigma**2))
    exponential = np.exp(-((x-nu)**2)/(2*sigma**2))
    
    
    return 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - nu)**2 / (2 * sigma**2) )



def multivariate_normal(x, d, mean, covariance):
    #X a random vector of size d
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
    
def random_sample(length):
    values = np.random.random_sample((length))
    return values

def uniform_random(length):
    values = np.random.uniform(low=0.0, high=1.0, size=length)
    return values

class MCMC:
    pass

def lognormal(sigma,mean,x):
    return lognorm.pdf(x,sigma,0,mean)
    



if __name__ == "__main__":
    print("loaded...")

