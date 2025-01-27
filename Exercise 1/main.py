#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, freqz, filtfilt, windows

from helpers import getOutputDir

# Signal parameters:
f0 = 1 * 1e6        # Signal frequncy in KHz
fs = 50 * 1e6       # Sampling frequency in KHz
T = 10 * 1e-6       # Time interval in ms
N = int(T * fs) + 1 # Number of samples

def calculateSnr(signal, noise):
    # SNR_est = mean(signal^2) / mean(noise^2)
    return np.mean(signal**2) / np.mean(noise**2)

def calculateSnrDb(signal, noise):
    # Decibel: SNR_est_dB = 10*log10(SNR_est)
    return 10 * np.log10(calculateSnr(signal, noise))

def generateUnitSignal():
    return np.sin(2 * np.pi * f0 * np.arange(0, T, 1/fs))

def generateUnitNoise():
    return np.random.normal(0, 1, N)

def generateTimeAxis():
    return np.arange(0, T, 1/fs)

def normalizeSignal(signal):
    return signal / np.max(np.abs(signal))

def scaleNoiseToTargetSNR(signal, noise, targetSNR):
    snrNoScaling = calculateSnr(signal, noise)
    # scale = sqrt(SNR_est / 10^(SNR/10))
    scale = np.sqrt(snrNoScaling / 10**(targetSNR/10)) 
    return noise * scale

def addPlot(ax, y, x=None,
            title = None, xLabel = None, yLabel = None,
            xLim = None, yLim = None,
            xticks = None, yticks = None,
            outputFile = None
            ):
    ax.xaxis.set_tick_params(labelsize=32, width=4)
    ax.yaxis.set_tick_params(labelsize=32, width=4)
    if x is not None:
        ax.plot(x, y, linewidth=4)
    else:
        ax.plot(y, linewidth=4)
    if title is not None:
        ax.set_title(title, fontsize=48)
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize=32)
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize=32)
    if xLim is not None:
        ax.set_xlim(xLim)
    if yLim is not None:
        ax.set_ylim(yLim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if outputFile is not None:
        saveFigure(outputFile, fig=ax.figure)

def saveFigure(fileName: str, fig = None):
    os.makedirs(f"{getOutputDir(1)}/figures", exist_ok=True)
    if fig is not None:
        fig.savefig(f"{getOutputDir(1)}/figures/{fileName}")
    else:
        plt.savefig(f"{getOutputDir(1)}/figures/{fileName}")


def task1(plot=True):

    def plotSignalNoiseCombined(subfig, time,
                                signal, signalAmplitude,
                                noise, noiseAmplitude,
                                title
                                ):
        subfig.suptitle(title, fontsize=36)

        axes = subfig.subplots(nrows=3, ncols=1, sharex=True)
        addPlot(axes[0], x=time, y=signal,
                title=f"Signal (A={signalAmplitude})", yLabel="Amplitude [A]")
        addPlot(axes[1], x=time, y=noise,
                title=f"Noise (A={noiseAmplitude})", yLabel="Amplitude [A]")
        addPlot(axes[2], x=time, y=signal + noise,
                title=f"Combined Signal + Noise", xLabel="Time [us]",
                yLabel="Amplitude [A]")

        return axes
    
    time = generateTimeAxis()
    unitSignal = generateUnitSignal()
    unitNoise = generateUnitNoise()

    signals = []
    noises = []

    signalAmplitudes = [1, 3, 2.5]
    noiseAmplitudes = [1, 1, 2]

    for sAmp, nAmp in zip(signalAmplitudes, noiseAmplitudes):
        signal = unitSignal * sAmp
        noise = unitNoise * nAmp
        signals.append(signal)
        noises.append(noise)
    
    if plot:
        # Plot unit signal/noise:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time*1e6, y=signals[0],
                title="Unit Signal", xLabel="Time [us]", yLabel="Amplitude [A]",
                outputFile="task1_signal.png")
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time*1e6, y=noises[0],
                title="Unit Noise", xLabel="Time [us]", yLabel="Amplitude [A]",
                outputFile="task1_noise.png")
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        addPlot(axes, x=time*1e6, y=signals[0] +noises[0],
                title="Unit Signal with Unit Noise",
                xLabel="Time [us]", yLabel="Amplitude [A]",
                outputFile="task1_combined.png")
    
        # Plot signal/noise with different amplitudes
        fig = plt.figure(figsize=(50, 40))
        subfigs = fig.subfigures(1, 3)
        for subfig, signal, sAmp, noise, nAmp in zip(subfigs,
                                                     signals, signalAmplitudes,
                                                     noises, noiseAmplitudes
        ):
            plotSignalNoiseCombined(subfig, time*1e6, signal,
                                    sAmp, noise, nAmp,
                                    f"Signal Amplitude: {sAmp}, Noise Amplitude: {nAmp}")
        
        subfigs[0].suptitle("Equal Signal- and Noise Amplitude\n Signal is somewhat visible",
                            fontsize=36)
        subfigs[1].suptitle("Greater Signal Amplitude\nSignal is clearly visible",
                            fontsize=36)
        subfigs[2].suptitle("Greater Noise Amplitude\nSignal is very hard to see",
                            fontsize=36)
        
        saveFigure("task1_all_plots.png", fig)

    return signals, noises

def task2(time, signals, noises, plot=True):
    # Calculate and show SNR:
    snrsdB = []
    for signal, noise in zip(signals, noises):
        snrsdB.append(calculateSnrDb(signal, noise))

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
        for ax, signal, noise, snr in zip(axes, signals, noises, snrsdB):
            addPlot(ax, x=time*1e6, y=signal + noise,
                    title=f"Signal + Noise (SNR: {snr:.2f} dB)",
                    xLabel="Time [us]", yLabel="Amplitude [A]"
            )
        saveFigure("task2_snr.png", fig)
    
    # Scale noise to try and reach target SNR, show results:
    TARGET_SNR_dB = 7

    unitSignal = signals[0]
    unitNoise = noises[0]

    snrNoScaling = calculateSnr(unitSignal, unitNoise)

    scale = np.sqrt(snrNoScaling / 10**(TARGET_SNR_dB/10))  # scale = sqrt(SNR_est / 10^(SNR/10))
    scaledNoise = unitNoise * scale

    snrScaledNoisedB = calculateSnrDb(unitSignal, scaledNoise)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
        title = f"Signal + Noise (SNR: {snrScaledNoisedB:.2f}dB, Target SNR: {TARGET_SNR_dB:.2f}dB)"
        addPlot(axes, x=time*1e6, y=normalizeSignal(signal + scaledNoise),
                title=title,
                xLabel="Time [us]", yLabel="Amplitude [A]"
        )
        saveFigure("task2_snr_scaled_normalized.png", fig)

    return unitSignal, scaledNoise

def task3(time, signal, noise, plot=True):

    def calculatePowerSpectrum(signal):
        # P(f) = 1/N * abs(fft(signal, N)^2
        N = len(signal)
        spectrum = np.fft.fft(signal, N)
        powerSpectrum = (1/N) * np.abs(spectrum)**2
        powerSpectrum = np.fft.fftshift(powerSpectrum)
        powerSpectrum = 20 * np.log10(powerSpectrum)
        freqs = np.fft.fftfreq(N, d=1/fs)
        freqs = np.fft.fftshift(freqs)
        return powerSpectrum, freqs
    
    def plotSpectrum(ax, spectrum, freqs, title):
        addPlot(ax, x=freqs*1e-6, y=spectrum, title=title,
                xLabel="Frequency [MHz]", yLabel="Amplitude [dB]",
                xLim=(-(fs*1e-6)/2, (fs*1e-6)/2)
        )

    signalSpectrum, signalFreqs = calculatePowerSpectrum(signal)
    noisySpectrum, noisyFreqs = calculatePowerSpectrum(normalizeSignal(signal + noise))

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 20))
        plotSpectrum(axes[0], signalSpectrum, signalFreqs,
                     "Signal Power Spectrum")
        plotSpectrum(axes[1], noisySpectrum, noisyFreqs,
                     "Signal + Noise Power Spectrum")
        saveFigure("task3_spectra.png", fig)
    
    return noisySpectrum, noisyFreqs

def task4(noisySpectrum, noisyFreqs, plot=True):
    def getFilterCoeffs(fLow, fHigh, filterOrder, gain):
        b, a = butter(filterOrder, [fLow, fHigh],
                         btype="bandpass", output="ba", fs=fs
        )
        b *= gain
        return b, a

    # Designing a bandpass filter and showing response:

    fLow = f0 * 0.9
    fHigh = f0 * 1.1
    filterOrder = 5
    gain = 30

    b, a = getFilterCoeffs(fLow, fHigh, filterOrder, gain)
    
    w, h = freqz(b, a, fs=fs, worN=np.linspace(-fs/2, fs/2, 1000))

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

        addPlot(axes, x=w*1e-6, y=np.abs(h), title="Filter Response",
                xLabel="Frequency [MHz]", yLabel="Response [dB]",
                xLim=(-(fs*1e-6)/2, (fs*1e-6)/2),
                # xticks=np.arange(-1, 10, 1),
                yLim=(-30, 40),
        )
        addPlot(axes, x=noisyFreqs*1e-6, y=noisySpectrum)
        axes.legend(["Filter Response", "Signal Power Spectrum"],
                    prop={"size": 36})
        saveFigure("task4_filter_response.png", fig)

    # Applying the filter:

    time = generateTimeAxis()        
    signal = generateUnitSignal()
    noise = scaleNoiseToTargetSNR(signal, generateUnitNoise(), -10)

    snrdBNoFilter = calculateSnrDb(signal, noise)

    combined = normalizeSignal(signal + noise)

    b, a = getFilterCoeffs(f0 * 0.8, f0 * 1.2, 6, 1)
    filtered = filtfilt(b, a, combined)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

    addPlot(axes, x=time*1e6, y=combined, title=f"Unfiltered vs. filtered signal",
            xLabel="Time [us]", yLabel="Amplitude [A]"
    )
    addPlot(axes, x=time*1e6, y=filtered)
    axes.legend([f"Signal + Noise (SNR: {snrdBNoFilter:.2f}dB)",
        f"Filtered Signal + Noise (SNR: {calculateSnrDb(signal, filtered):.2f}dB)"
        ], prop={"size": 36}
    )
    saveFigure("task4_filtered_signal.png", fig)

    # Apply a Hamming window before filtering:

    window = windows.hamming(N)
    windowedSignal = combined * window
    filteredWindowed = filtfilt(b, a, windowedSignal)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

    addPlot(axes, x=time*1e6, y=combined, title=f"Unfiltered vs. filtered+windowed signal",
            xLabel="Time [us]", yLabel="Amplitude [A]"
    )
    addPlot(axes, x=time*1e6, y=filteredWindowed)
    axes.legend([f"Noisy signal (SNR: {snrdBNoFilter:.2f}dB)",
        f"Windowed, filtered signal (SNR: {calculateSnrDb(signal, filteredWindowed):.2f}dB)"
        ], prop={"size": 36}
    )
    saveFigure("task4_filtered_windowed_signal.png", fig)

def main():
    time = generateTimeAxis()

    signals, noises = task1(plot=True)

    signal, scaledNoise = task2(time, signals, noises, plot=True)

    noisySpectrum, noisyFreqs = task3(time, signal, scaledNoise, plot=True)

    task4(noisySpectrum, noisyFreqs, plot=True)
    # plt.show()

if __name__ == '__main__':
    main()
