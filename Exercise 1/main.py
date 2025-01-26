#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from helpers import getOutputDir

# Signal parameters:
f0 = 1      # Signal frequncy in MHz
fs = 50     # Sampling frequency in MHz
T = 10      # Time interval in us

def calculateSnr(signal, noise):
    # SNR_est = mean(signal^2) / mean(noise^2)
    return np.mean(signal**2) / np.mean(noise**2)

def calculateSnrDb(signal, noise):
    # Decibel: SNR_est_dB = 10*log10(SNR_est) [dB]
    return 10 * np.log10(calculateSnr(signal, noise))

def generateUnitSignal():
    return np.sin(2 * np.pi * f0 * np.arange(0, T, 1/fs))

def generateUnitNoise():
    return np.random.normal(0, 1, len(np.arange(0, T, 1/fs)))

def generateTimeAxis():
    return np.arange(0, T, 1/fs)

def addPlot(ax, y, x=None, title = None, xLabel = None, yLabel = None, xLim = None, yLim = None, outputFile = None):
    ax.xaxis.set_tick_params(labelsize=32, width=4)
    ax.yaxis.set_tick_params(labelsize=32, width=4)
    if x is not None:
        ax.plot(x, y, linewidth=4)
    else:
        ax.plot(y, linewidth=4)
    if(title):
        ax.set_title(title, fontsize=48)
    if xLabel:
        ax.set_xlabel(xLabel, fontsize=32)
    if yLabel:
        ax.set_ylabel(yLabel, fontsize=32)
    if xLim:
        ax.set_xlim(xLim)
    if yLim:
        ax.set_ylim(yLim)
    if outputFile:
        plt.savefig(f"{getOutputDir(1)}/{outputFile}")

def task1(time, unitSignal, unitNoise, plot=True):

    def plotSignalNoiseCombined(subfig, time, signal, signalAmplitude, noise, noiseAmplitude, title):
        subfig.suptitle(title, fontsize=36)

        axes = subfig.subplots(nrows=3, ncols=1, sharex=True)
        addPlot(axes[0], x=time, y=signal, title=f"Signal (A={signalAmplitude})", yLabel="Amplitude [A]")
        addPlot(axes[1], x=time, y=noise, title=f"Noise (A={noiseAmplitude})", yLabel="Amplitude [A]")
        addPlot(axes[2], x=time, y=signal + noise, title=f"Combined Signal + Noise", xLabel="Time [us]", yLabel="Amplitude [A]")

        return axes
    
    signals = []
    noises = []

    signalAmplitudes = [1, 3, 2.5]
    noiseAmplitudes = [1, 2, 2]

    for sAmp, nAmp in zip(signalAmplitudes, noiseAmplitudes):
        signal = unitSignal * sAmp
        noise = unitNoise * nAmp
        signals.append(signal)
        noises.append(noise)
    
    if plot:
        # Plot unit signal/noise:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time, y=signals[0], title="Unit Signal", xLabel="Time [us]", yLabel="Amplitude [A]", outputFile="task1_signal.png")
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time, y=noises[0], title="Unit Noise", xLabel="Time [us]", yLabel="Amplitude [A]", outputFile="task1_noise.png")
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time, y=signals[0] +noises[0], title="Unit Signal with Unit Noise", xLabel="Time [us]", yLabel="Amplitude [A]", outputFile="task1_combined.png")
    
        # Plot signal/noise with different amplitudes
        fig = plt.figure(figsize=(50, 20))
        subfigs = fig.subfigures(1, 3)
        for subfig, sAmp, nAmp in zip(subfigs, signalAmplitudes, noiseAmplitudes):
            plotSignalNoiseCombined(subfig, time, signal, sAmp, noise, nAmp, f"Signal Amplitude: {sAmp}, Noise Amplitude: {nAmp}")
        
        subfigs[0].suptitle("Equal Signal- and Noise Amplitude\n Signal is somewhat visible", fontsize=36)
        subfigs[1].suptitle("Greater Signal Amplitude\nSignal is clearly visible", fontsize=36)
        subfigs[2].suptitle("Greater Noise Amplitude\nSignal is hard to see", fontsize=36)
        
        fig.savefig(f"{getOutputDir(1)}/task1_all_plots.png")

    return signals, noises

def task2(time, signals, noises, plot=True):

    def plotCombinedSNR(ax, time, signal, noise, snrDb, targetSnrDb=None):
        if targetSnrDb is not None:
            title = f"Signal + Noise (SNR: {snrDb:.2f}dB, Target SNR: {targetSnrDb:.2f}dB)"
        else:
            title = f"Signal + Noise (SNR: {snrDb:.2f} dB)"
        addPlot(ax, x=time, y=signal + noise, title=title, xLabel="Time [us]", yLabel="Amplitude [A]")

    # Calculate and show SNR:
    snrs = []
    for signal, noise in zip(signals, noises):
        snrs.append(calculateSnrDb(signal, noise))

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
        for ax, signal, noise, snr in zip(axes, signals, noises, snrs):
            plotCombinedSNR(ax, time, signal, noise, snr)

        fig.savefig(f"{getOutputDir(1)}/task2_snr.png")            
    
    # Scale noise to try and reach target SNR, show results:
    TARGET_SNR_dB = 7

    unitSignal = signals[0]
    unitNoise = noises[0]

    snrNoScaling = calculateSnr(unitSignal, unitNoise)

    scale = np.sqrt(snrNoScaling / 10**(TARGET_SNR_dB/10))  # scale = sqrt(SNR_est / 10^(SNR/10))
    scaledNoise = unitNoise * scale

    snrScaledNoise = calculateSnrDb(unitSignal, scaledNoise)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
        plotCombinedSNR(axes, time, unitSignal, noise, snrScaledNoise, TARGET_SNR_dB)
        fig.savefig(f"{getOutputDir(1)}/task2_snr_scaled.png")

    return snrScaledNoise    

def task3(time, signal, noise, plot=True):
    """
        Task 3 - Power spectrum (frequency) analysis
        1. Calculate the power spectrum Ps(f) of the continuous sinusoidal signal, and Psn(f) of
            the signal + noise.
        2. Plot the frequency spectrum of the signal and the signal + noise in diberent (sub)plots.
    """
    def calculatePowerSpectrum(signal):
        # P(f) = 1/N*abs(fft(signal, N)^2
        # Where N is the number of frequency components, here distributed
        # between [-Fs/2, Fs/2].
        N = len(signal)
        spectrum = np.fft.fft(signal)
        spectrum = np.fft.fftshift(spectrum)
        powerSpectrum = (1/N) * np.abs(spectrum)**2
        freqs = np.fft.fftfreq(N, d=1/fs)
        return powerSpectrum, freqs
    
    def plotSpectrum(ax, spectrum, freqs, title):
        idx = np.argsort(freqs)
        addPlot(ax, x=freqs, y=10 * np.log10(spectrum), title=title, xLabel="Frequency [MHz]", yLabel="Amplitude [A]", xLim=(-fs/2, fs/2))
    

    signalSpectrum, signalFreqs = calculatePowerSpectrum(signal)
    noisySpectrum, noisyFreqs = calculatePowerSpectrum(signal + noise)

    if plot:
        _, axes = plt.subplots(nrows=1, ncols=2)
        plotSpectrum(axes[0], signalSpectrum, signalFreqs, "Signal Power Spectrum")
        plotSpectrum(axes[1], noisySpectrum, noisyFreqs, "Signal + Noise Power Spectrum")
    
    return signalSpectrum, signalFreqs, noisySpectrum, noisyFreqs

def task4(signalSpectrum, signalFreqs, plot=True):
    """
        Task 4 - Filtering to improve the signal-to-noise ratio
        1. Design a band pass filter (IIR or FIR type) to keep the signal of interest but
            also attenuate other noise components.
        2. 2. Plot the frequency response of the filter,
            …Maybe even on top of the spectrum of the signal or signal+noise.
        3. Calculate the SNR and compare to the ideal value calculated previously.
        4. Do the procedure for the high and low SNR scenarios. Are you able to reconstruct the
            signal? Try to do as good as possible by tuning the parameters of the filter (low
            and high filter frequencies, and filter order).
    """
    # Create a band pass filter
    fLow = f0 - (f0*0.1)
    fHigh = f0 + (f0*0.1)
    filterOrder = 3

    b, a = signal.butter(filterOrder, [fLow, fHigh], btype="bandpass", output="ba", fs=fs)

    if plot:
        _, axes = plt.subplots(nrows=1, ncols=2)
        w, h = signal.freqs(b, a)
        responseDb = 10*np.log10(np.abs(h))

        #Find overlapping part of the frequency response and the signal spectrum
        overlap = np.intersect1d(signalFreqs, w)
        indices = np.where(np.logical_and(signalFreqs > min(responseDb), signalFreqs < max(responseDb)), signalFreqs, responseDb)

        addPlot(axes[0], x=signalFreqs, y=signalSpectrum[indices], title="Signal Spectrum", xLabel="Frequency [MHz]", yLabel="Amplitude [A]")
        
        # addPlot(axes[0], x=w, y=overlap, title="Frequency Response", xLabel="Frequency [MHz]", yLabel="Amplitude [dB]")
        # addPlot(axes[1], x=signalFreqs, y=responseDb)[np.indices(responseDb == 0)], title="Frequency Response", xLabel="Frequency [MHz]", yLabel="Amplitude [dB]")
        # addPlot(axes[1], x=signalFreqs, y=signalSpectrum)




def bonusTask():
    """
        Bonus
        Set the SNR to very low so that the signal is barely visible or not at all. Try to find
        a way to improve results of the reconstructed signal. Calculate the SNR and compare to
        the values calculated previously as a metric in addition to the visual comparison. Use
        any source to help you and have fun.
    """
    pass

def main():
    time = generateTimeAxis()
    unitSignal = generateUnitSignal()
    unitNoise = generateUnitNoise()

    signals, noises = task1(time, unitSignal, unitNoise, plot=True)
    scaledNoise = task2(time, signals, noises, plot=True)
    # signalSpectrum, signalFreqs, noisySpectrum, noisyFreqs = task3(time, signals[0], scaledNoise, plot=False)
    # task4(signalSpectrum, signalFreqs, plot=True)
    # bonusTask()
    # plt.show()

if __name__ == '__main__':
    main()
