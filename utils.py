import numpy as np
import matplotlib.pyplot as plt
from attributedict.collections import AttributeDict
from numpy.fft import fft, ifft
from scipy.interpolate import interp1d

def plot_simEEG(*args):
    """
    plot_simEEG - plot function for MXC's course on neural time series analysis
    INPUTS:  EEG : eeglab structure
             chan : channel to plot (default = 0)
           fignum : figure to plot into (default = 0)
    """
    if not args:
        raise ValueError('No inputs!')
    elif len(args) == 1:
        EEG = args[0]
        chan, fignum = 0, 0
    elif len(args) == 2:
        EEG, chan = args
        fignum = 0
    elif len(args) == 3:
        EEG, chan, fignum = args

    plt.figure(fignum, figsize=(16,10))
    plt.clf()

    # ERP
    plt.subplot(211)
    plt.plot(EEG.times, np.squeeze(EEG.data[chan,:,:]), linewidth=0.5, color=[.75, .75, .75])
    plt.plot(EEG.times, np.squeeze(np.mean(EEG.data[chan,:,:], axis=1)), 'k', linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    plt.title(f'ERP from channel {chan}')

    # static power spectrum
    hz = np.linspace(0, EEG.srate, EEG.pnts)
    if len(EEG.data.shape) == 3:
        pw = np.mean((2 * np.abs(fft(EEG.data[chan,:,:], axis=0) / EEG.pnts))**2, axis=1)
    else:
        pw = (2 * np.abs(fft(EEG.data[chan,:], axis=0) / EEG.pnts))**2

    plt.subplot(223)
    plt.plot(hz, pw, linewidth=2)
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Static power spectrum')
    
    # time-frequency analysis
    frex = np.linspace(2, 30, 40)  # frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex))**2  # number of wavelet cycles (hard-coded to 3 to 10)

    # setup wavelet and convolution parameters
    wavet = np.arange(-2, 2, 1/EEG.srate)
    halfw = len(wavet) // 2
    nConv = EEG.pnts * EEG.trials + len(wavet) - 1

    # initialize time-frequency matrix
    tf = np.zeros((len(frex), EEG.pnts))

    # spectrum of data
    dataX = fft(np.reshape(EEG.data[chan,:,:], -1, order='F'), n=nConv)
    # loop over frequencies
    for fi in range(len(frex)):
        # create wavelet
        waveX = fft(np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-wavet**2 / waves[fi]), n=nConv)
        waveX = waveX / np.max(waveX) # normalize
        
        # convolve
        as_ = ifft(waveX * dataX)
        # trim and reshape
        as_ = np.reshape(as_[halfw:len(as_)-halfw+1], [EEG.pnts, EEG.trials], order='F')

        # power
        tf[fi, :] = np.mean(np.abs(as_), axis=1)

    # show a map of the time-frequency power
    plt.subplot(224)
    plt.contourf(EEG.times, frex, tf, 40, cmap='jet')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency plot')

    plt.show()
