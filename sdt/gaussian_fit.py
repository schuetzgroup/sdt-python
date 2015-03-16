#!/usr/bin/env python
# This script is to fit 1d or 2d gaussian profile on a 1d or 2d grid data.
# 2d gaussian fit functions are written similar to to https://gist.github.com/andrewgiessel/6122739
# source: https://github.com/indiajoe/HandyTools4Astronomers.git


from __future__ import division
import numpy as np
import numpy.ma
import scipy.optimize

#------------------------------------1D gaussian fitting functions...
def moments1D(inpData):
    """ Returns the (amplitude, center,sigma, bkgC, bkgSlope) estimated from moments in the 1d input array inpData """

    bkgC=inpData[0]  #Taking first point as intercept and fitting a straght line for slope
    bkgSlope=(inpData[-1]-inpData[0])/(len(inpData)-1.0)

    Data=np.ma.masked_less(inpData-(bkgC+bkgSlope*np.arange(len(inpData))),0)   #Removing the background for calculating moments of pure gaussian
    #We also masked any negative values before measuring moments

    amplitude=Data.max()

    total=float(Data.sum())
    Xcoords=np.arange(Data.shape[0])

    center=(Xcoords*Data).sum()/total

    sigma=np.sqrt(np.ma.sum((Data*(Xcoords-center)**2))/total)

    return amplitude,center,sigma,bkgC,bkgSlope

def Gaussian1D(amplitude, center,sigma,bkgC,bkgSlope):
    """ Returns a 1D Gaussian function with input parameters. """
    Xc=center  #Center
    #Now lets define the 1D gaussian function
    def Gauss1D(x) :
        """ Returns the values of the defined 1d gaussian at x """
        return amplitude*np.exp(-(((x-Xc)/sigma)**2)/2) +bkgC+bkgSlope*x

    return Gauss1D

def FitGauss1D(Data,ip=None):
    """ Fits 1D gaussian to Data with optional Initial conditions ip=(amplitude, center, sigma, bkgC, bkgSlope)
    Example:
    >>> X=np.arange(40,dtype=np.float)
    >>> Data=np.exp(-(((X-25)/5)**2)/2) +1+X*0.5
    >>> FitGauss1D(Data)
    (array([  1. ,  25. ,   5. ,   1. ,   0.5]), 2)
     """
    if ip is None:   #Estimate the initial parameters from moments
        ip=moments1D(Data)

    def errfun(ip):
        return np.ravel(Gaussian1D(*ip)(*np.indices(Data.shape)) - Data)

    p, success = scipy.optimize.leastsq(errfun, ip)

    return p,success


#------------------------------------------------------------#
#------------------------------------2D gaussian fitting functions...
def moments2D(inpData):
    """ Returns the (amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e) estimated from moments in the 2d input array Data """

    bkg=np.median(np.hstack((inpData[0,:],inpData[-1,:],inpData[:,0],inpData[:,-1])))  #Taking median of the 4 edges points as background
    Data=np.ma.masked_less(inpData-bkg,0)   #Removing the background for calculating moments of pure 2D gaussian
    #We also masked any negative values before measuring moments

    amplitude=Data.max()

    total= float(Data.sum())
    Xcoords,Ycoords= np.indices(Data.shape)

    xcenter= (Xcoords*Data).sum()/total
    ycenter= (Ycoords*Data).sum()/total

    RowCut= Data[int(xcenter),:]  # Cut along the row of data near center of gaussian
    ColumnCut= Data[:,int(ycenter)]  # Cut along the column of data near center of gaussian
    xsigma= np.sqrt(np.ma.sum(ColumnCut* (np.arange(len(ColumnCut))-xcenter)**2)/ColumnCut.sum())
    ysigma= np.sqrt(np.ma.sum(RowCut* (np.arange(len(RowCut))-ycenter)**2)/RowCut.sum())

    #Ellipcity and position angle calculation
    Mxx= np.ma.sum((Xcoords-xcenter)*(Xcoords-xcenter) * Data) /total
    Myy= np.ma.sum((Ycoords-ycenter)*(Ycoords-ycenter) * Data) /total
    Mxy= np.ma.sum((Xcoords-xcenter)*(Ycoords-ycenter) * Data) /total
    e= np.sqrt((Mxx - Myy)**2 + (2*Mxy)**2) / (Mxx + Myy)
    pa= 0.5 * np.arctan(2*Mxy / (Mxx - Myy))
    rot= np.rad2deg(pa)

    return amplitude,xcenter,ycenter,xsigma,ysigma, rot,bkg, e

def Gaussian2D(amplitude, xcenter, ycenter, xsigma, ysigma, rot,bkg):
    """ Returns a 2D Gaussian function with input parameters. rotation input rot should be in degress """
    rot=np.deg2rad(rot)  #Converting to radians
    Xc=xcenter*np.cos(rot) - ycenter*np.sin(rot)  #Centers in rotated coordinates
    Yc=xcenter*np.sin(rot) + ycenter*np.cos(rot)
    #Now lets define the 2D gaussian function
    def Gauss2D(x,y) :
        """ Returns the values of the defined 2d gaussian at x,y """
        xr=x * np.cos(rot) - y * np.sin(rot)  #X position in rotated coordinates
        yr=x * np.sin(rot) + y * np.cos(rot)
        return amplitude*np.exp(-(((xr-Xc)/xsigma)**2 +((yr-Yc)/ysigma)**2)/2) +bkg

    return Gauss2D

def FitGauss2D(Data,ip=None):
    """ Fits 2D gaussian to Data with optional Initial conditions ip=(amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg)
    Example:
    >>> X,Y=np.indices((40,40),dtype=np.float)
    >>> Data=np.exp(-(((X-25)/5)**2 +((Y-15)/10)**2)/2) + 1
    >>> FitGauss2D(Data)
    (array([  1.00000000e+00,   2.50000000e+01,   1.50000000e+01, 5.00000000e+00,   1.00000000e+01,   2.09859373e-07, 1]), 2)
     """
    if ip is None:   #Estimate the initial parameters form moments and also set rot angle to be 0
        ip=moments2D(Data)[:-1]   #Remove ellipticity from the end in parameter list

    Xcoords,Ycoords= np.indices(Data.shape)
    def errfun(ip):
        dXcoords= Xcoords-ip[1]
        dYcoords= Ycoords-ip[2]
        Weights=np.sqrt(np.square(dXcoords)+np.square(dYcoords)) # Taking radius as the weights for least square fitting
        return np.ravel((Gaussian2D(*ip)(*np.indices(Data.shape)) - Data)/np.sqrt(Weights))  #Taking a sqrt(weight) here so that while scipy takes square of this array it will become 1/r weight.

    p, success = scipy.optimize.leastsq(errfun, ip)

    return p,success

