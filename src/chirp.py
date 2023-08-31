import numpy as np
import numpy.matlib as mat
import math
from numpy.core.function_base import linspace


def linearChirp(init_freq, fin_freq, T, x, noise):
    """
    Creates Linear Chrip signal 

    Parameters
    ----------
    init_freq : int
        starting frequency 

    fin_freq : int
        ending frequency

    T : int 
        time it takes to go from init_freq to fin_freq 

    x : numpy array 
        linearly spaced numpy array time vector 

    Returns
    -------
    numpy array of signal, numpy array of frequency
    """

    f0 = init_freq
    f1 = fin_freq
    c = (f1-f0)/T
    y = np.sin(2*np.pi*(c/2 * x**2 + f0*x))
    ft = c*x+f0
    if noise:
        y += np.random.normal(0, 1)

    return y, ft


def geometricChirp(init_freq, fin_freq, T, x, noise):
    """
    Creates Geometric Chrip signal 

    Parameters
    ----------
    init_freq : int
        starting frequency 

    fin_freq : int
        ending frequency

    T : int 
        time it takes to go from init_freq to fin_freq 

    x : numpy array 
        linearly spaced numpy array time vector 

    Returns
    -------
    numpy array of signal, numpy array of frequency
    """

    f0 = init_freq
    f1 = fin_freq
    k = (f1/f0)**(1/T)
    ft_geo = f0*k**x
    frac = (k**x - 1)/np.log(k)
    y_geo = np.sin(2*np.pi*f0*(frac))

    if noise:
        y_geo += np.random.normal(0, 1)
    return y_geo, ft_geo


def sinusoidalMod(carrier_freq, mod_freq, x, noise):
    """
    Creates Sinusoidal Modulated signal 

    Parameters
    ----------
    carrier_freq : int
        carrier frequency 

    mod_freq : int
        modulated frequency

    x : numpy array 
        linearly spaced numpy array time vector 

    Returns
    -------
    numpy array of signal, numpy array of frequency
    """
    y = np.sin(carrier_freq*x) * np.sin(mod_freq*x)
    ft = np.sin(carrier_freq*x)
    if noise:
        y += np.random.normal(0, 1)

    return y, ft

# Print iterations progress


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "linearChirp":
    linearChirp()

if __name__ == "geometricChirp":
    geometricChirp()

if __name__ == "sinusoidalMod":
    sinusoidalMod()
if __name__ == "printProgressBar":
    printProgressBar()
