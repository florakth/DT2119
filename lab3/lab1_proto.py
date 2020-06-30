
# DT2119, Lab 1 Feature Extraction

import numpy as np
from lab1_tools import *
from scipy import signal,fftpack
from scipy.fftpack.realtransforms import dct

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    frame = np.array(samples[0:winlen].reshape((1, winlen)))
    stepsize = winlen - winshift
    for i in range(stepsize, len(samples) - winlen, stepsize):
        frame = np.vstack((frame, samples[i:i+winlen].reshape((1, winlen))))
    return frame
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    a = [1]
    b = [1, -p]
    return signal.lfilter(b, a, input, zi=None)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    window = signal.hamming(input.shape[1], sym=False)
    #plt.plot(window)
    #plt.title('Hamming window')
    #plt.show()
    output = []
    for frame in input:
        output.append(frame * window)
    return output


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    spec = []
    for frame in input:
        spec.append(np.square(np.abs(fftpack.fft(frame, nfft))))
    return spec

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = np.array(input).shape[1]
    mel = trfbank(samplingrate, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, 
                  nlogfilt=27, equalareas=False)
    output = np.dot(input, mel.T)
    output = np.where(output == 0.0, np.finfo(float).eps, output) 
    return np.log(output)

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    #return dct(input, type=2, axis=1, norm = 'ortho')[:,: nceps]
    return dct(input)[:,: nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    N = x.shape[0]
    M = y.shape[0]
    global_dist = 0
    LD = np.zeros((N, M))
    AD = np.zeros((N, M)) 
    path_mat = []

    for i in range(N):
        for j in range(M):
            LD[i, j] = dist(x[i], y[j])
    
    nrows, ncols = AD.shape
    # SET column 0
    for row in range(1,nrows):
        AD[row,0] = AD[row-1,0] + LD[row,0]
    # SET row 0
    for col in range(1,ncols):
        AD[0,col] = AD[0,col-1] + LD[0,col]

    for row in range(1, nrows): # Start from 1 to avoid out of bounds
        for col in range(1, ncols):
            minimum_dist = LD[row, col] + min(AD[row, col-1], AD[row-1, col], AD[row-1, col-1]) 
            AD[row, col] = minimum_dist
  
    backtracking = True
    i, j = nrows-1, ncols-1
    path_mat.append((i,j))
    while backtracking:
        min_dist =  min(AD[i, j-1],
                        AD[i-1, j],
                        AD[i-1, j-1])
        min_idx = np.where(AD==min_dist)
        i, j = min_idx[0][0], min_idx[1][0]
        path_mat.append((i,j))
        if i == 0 and j == 0:
            backtracking = False

    global_dist = AD[nrows-1,ncols-1] / (N + M)
        
    return global_dist, LD, AD, path_mat

def Euclidean(x,y):
    return np.linalg.norm(x-y)

def backtrack(AD):
    """
    Returns best path through accumulated distance matrix AD
    """
    N = AD.shape[0]
    M = AD.shape[1]
    path = [(N - 1, M - 1)]
    i = N - 1
    j = M - 1
    while(i > 0 or j > 0):
        if( i > 0 and j > 0):
            argmin = np.argmin([AD[i - 1, j - 1], AD[i - 1, j], AD[i, j - 1]])
            if(argmin == 0):
                path.append((i - 1, j - 1))
                i = i - 1
                j = j - 1
            elif(argmin == 1):
                path.append((i - 1, j))
                i = i - 1
            elif(argmin == 2):
                path.append((i, j - 1))
                j = j - 1
        elif(i == 0 and j > 0):
            path.append((0, j - 1))
            j = j - 1
        else:
            path.append((i - 1, 0))
            i = i - 1
    return path

