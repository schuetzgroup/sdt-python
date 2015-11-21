"""Easy fitting with Gaussian functions

This module allows for fitting Gaussian functions to 1D or 2D data. It is
based on https://gist.github.com/andrewgiessel/6122739.

Examples
--------
>>> data = np.load("gaussian_2d.npy")
>>> params = fit(data)  # get parameters of the fitted Gaussian
>>> fitted_gaussian = gaussian(**params)  # actually construct the Gaussian
>>> fitted_gaussian(2, 3)  # get values at coordinates (2, 3)
15.0
"""
import numpy as np
import scipy.optimize

def guess_paramaters(data):
    """Initial guess of parameters of the Gaussian

    This function does a crude estimation of the parameters:
    - The background is guessed by looking at the edges of `data`.
    - The center of the Gaussian is approximated by the center of mass.
    - sigma as well as the angle of rotation are estimated by looking at second
      moments
    - The amplitude is taken as the maximum above background

    Parameters
    ----------
    data : numpy.ndarray
        Data for which the parameter should be guessed

    Returns
    -------
    dict
        This dict has the follwing keys:
        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the guessed center
        - "sigma" (np.ndarray)
        - "background" (float)
        - "angle" (float): guessed angle of rotation. Works (currently) only
          for 2D data.
        The keys match the arguments of `gaussian` so that the dict can be
        passed directly to the function.
    """
    # median of the edges as an estimate for the background
    interior_slice = slice(1, -1)
    edge_mask = np.ones(data.shape, dtype=np.bool)
    edge_mask[[interior_slice]*data.ndim] = False
    bg = np.median(data[edge_mask])

    # subtract background for calculation of moments, mask negative values
    data_bg = np.ma.masked_less(data - bg, 0)
    data_bg_sum = np.sum(data_bg)

    # maximum as an estimate for the amplitude
    amp = np.max(data_bg)

    # calculate 1st moments as estimates for centers
    all_idx = np.indices(data.shape)
    center = np.fromiter((np.sum(i * data_bg)/data_bg_sum for i in all_idx),
                         dtype=np.float)

    # calculate 2nd moments along lines through center as estimates for sigma
    all_slice = slice(0, None)  # selects all
    sigma = []
    for i, ci in enumerate(center):
        c = center.astype(np.int).tolist()  # indices closest to the center
        c[i] = all_slice  # only along the i-th coordinate select whole line
        line = data_bg[tuple(c)]
        idx = np.arange(len(line))
        sigma.append(np.sqrt(np.sum((line*(idx - ci)**2))/np.sum(line)))
    sigma = np.array(sigma)

    # for 2D data only: Use inertia matrix to determine angle of rotation
    angle = 0.
    if data.ndim == 2:
        m_xx = np.sum(data_bg*(all_idx[0]-center[0])**2) / data_bg_sum
        m_xy = (np.sum(data_bg*(all_idx[0]-center[0])*(all_idx[1]-center[1])) /
                data_bg_sum)
        m_yy = np.sum(data_bg*(all_idx[1]-center[1])**2) / data_bg_sum
        angle = 0.5 * np.arctan(2*m_xy / (m_xx - m_yy))

    return dict(amplitude=amp, center=center, sigma=sigma, background=bg,
                angle=angle)


def gaussian(amplitude, center, sigma, background, angle=0.):
    """Create a Gaussian function from parameters

    This returns a callable that computes Gaussian function values from
    coordinates according to the parameters supplied here. Currently, only
    1D and 2D Gaussians can be created.

    Parameters
    ----------
    amplitude : float
        amplitude of the Gaussian
    center : numpy.ndarray
        center coordinates. From the number of coordinates the dimensionality
        of the Gaussian is determined. Currently only 1D and 2D are supported.
    sigma : float or numpy.ndarray
        sigma of the Gaussian. If it is a float, the same sigma for all
        coordinate axes is assumed. If it is an array, its dimensions have to
        match `center` and the i-th element is interpreted as the sigma in
        the i-th coordinate axis.
    background : float
        additive offset
    angle : float, optional
        angle of rotation. Only applies to 2D Gaussians. Defaults to 0.

    Returns
    -------
    gauss : callable
        Takes as many arguments as center coordinates were supplied. The
        arguments need to be numpy.ndarrays, e. g. the results of a call to
        `numpy.indices`.

    Raises
    ------
    ValueError
        if either sigma has the wrong length or it was attempted to created
        a Gaussian of unsupported dimensions.
    """

    ndim = len(center)
    if np.isscalar(sigma):
        sigma = [sigma]*ndim
    if len(sigma) != ndim:
        raise ValueError("sigma needs to be a number or the number of sigma "
                         "values needs to match the number of center entries.")

    if ndim == 1:
        def gaussian_1d(x):
            """1D Gaussian created using `gaussian`"""
            return (amplitude * np.exp(-((x - center[0])/sigma[0])**2/2.) +
                    background)
        return gaussian_1d
    elif ndim == 2:
        cs = np.cos(angle)  # cache
        sn = np.sin(angle)  # cache

        def rotate(x, y):
            return x*cs - y*sn, x*sn + y*cs

        xc_r, yc_r = rotate(center[0], center[1])  # center coordinates rotated
        def gaussian2d(x, y):
            """2D Gaussian created using `gaussian`"""
            x_r, y_r = rotate(x, y)  # coordinates rotated
            arg = ((x_r - xc_r)/sigma[0])**2 + ((y_r - yc_r)/sigma[1])**2
            return amplitude * np.exp(-arg/2.) + background
        return gaussian2d
    else:
        raise ValueError("Only 1D and 2D Gaussians are supported at the"
                         "moment")


def fit(data, initial_guess=None):
    """Fit a Gaussian function to `data`

    Parameters
    ----------
    data : numpy.ndarray
        Data to which the Gaussian should be fitted
    initial_guess : dict, optional
        Initial guess for the parameters. The dict has the same keys and values
        as the one returned by `initial_guess` (and this function as well) so
        that it can be passed to `gaussian`. If None (the default), call
        `initial_guess` to determine initial values.

    Returns
    -------
    dict
        The dict contains the fitted parameters of the Gaussian. It dict has
        the follwing keys:
        - "amplitude" (float)
        - "center" (np.ndarray): coordinates of the center
        - "sigma" (np.ndarray)
        - "background" (float)
        - "angle" (float): angle of rotation. Works (currently) only for 2D
          data.
        The keys match the arguments of `gaussian` so that the dict can be
        passed directly to the function.

    Raises
    ------
    RuntimeError
        if the fit did not converge.
    """
    if initial_guess is None:
        initial_guess = guess_paramaters(data)

    all_idx = np.indices(data.shape)
    ndim = data.ndim

    def param_tuple_to_dict(params):
        return dict(amplitude=params[0], center=np.array(params[1:ndim+1]),
                    sigma=np.array(params[ndim+1:2*ndim+1]),
                    background=params[2*ndim + 1], angle=params[2*ndim + 2])

    def err_func(params):
        params = param_tuple_to_dict(params)
        curr_gauss = gaussian(**params)
        err = curr_gauss(*all_idx) - data
        # alternative: weigh the error
        return np.ravel(err)

    # scipy.optimize.leastsq expects a list of parameters. For this, flatten
    # the parameters structure
    initial_list = ([initial_guess["amplitude"]] +
                    initial_guess["center"].tolist() +
                    initial_guess["sigma"].tolist() +
                    [initial_guess["background"], initial_guess["angle"]])
    p, success = scipy.optimize.leastsq(err_func, initial_list)

    if success > 2:
        raise RuntimeError("Fit did not converge.")
    return param_tuple_to_dict(p)

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

    p, success = scipy.optimize.leastsq(errfun, ip, maxfev=1)

    return p,success


#------------------------------------------------------------#
#------------------------------------2D gaussian fitting functions...
def moments2D(inpData):
    """ Returns the (amplitude, xcenter, ycenter, xsigma, ysigma, rot, bkg, e) estimated from moments in the 2d input array Data """

    #bkg=np.median(np.hstack((inpData[0,:],inpData[-1,:],inpData[:,0],inpData[:,-1])))  #Taking median of the 4 edges points as background
    interior_slice = slice(1, -1)
    edge_mask = np.ones(inpData.shape, dtype=np.bool)
    edge_mask[[interior_slice]*inpData.ndim] = False
    bkg = np.median(inpData[edge_mask])
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

