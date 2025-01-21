import numpy as np
import matplotlib.pyplot as plt

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

def generateSignalAndNoise(signalAmplitude, noiseAmplitude):
    time = np.arange(0, T, 1/fs)
    signal = signalAmplitude * np.sin(2 * np.pi * f0 * time)
    noise = np.random.normal(0, noiseAmplitude, len(time))

    return time, signal, noise

def addPlot(ax, time, signal, title = None, xLabel = None, yLabel = None, xLim = None, yLim = None):
    ax.plot(time, signal)
    if(title):
        ax.set_title(title)
    if xLabel:
        ax.set_xlabel(xLabel)
    if yLabel:
        ax.set_ylabel(yLabel)
    if xLim:
        ax.set_xlim(xLim)
    if yLim:
        ax.set_ylim(yLim)

def task1():
    """
        Task 1 - Signal generation and plotting
        1.  Generate a continuous sinusoidal signal over time and plot it in a figure with labels.
        2. Generate random Gaussian noise of the same size and sampled at the same rate as the
            signal in (1) and plot it in a figure with labels.
        3. Combine the signal and noise by adding them together and plot the result in a figure.
            Vary the signal / noise amplitude to find two scenarios, a) one where the signal is clearly
            visible with only some noise, and b) one where the signal is clearly buried in noise,
            barely visible.
    """

    def plotSignalAndNoise(subfig, time, signal, signalAmplitude, noise, noiseAmplitude, title):
        subfig.suptitle(title)

        axes = subfig.subplots(nrows=3, ncols=1, sharex=True)

        for ax in axes:
            ax.set_xlabel("Time [us]")
            ax.set_ylabel("Amplitude [A]")

        axes[0].plot(time, signal)
        axes[0].set_title(f"Signal (A={signalAmplitude})")

        axes[1].plot(time, noise)
        axes[1].set_title(f"Noise (A={noiseAmplitude})")

        axes[2].plot(time, signal + noise)
        axes[2].set_title(f"Combined Signal + Noise")

        return axes
    
    
    fig = plt.figure()
    # fig.set_layout_engine("tight")
    subfigs = fig.subfigures(1, 3)
    signals = []
    noises = []

    sAmp, nAmp = 1, 1
    time, signal, noise = generateSignalAndNoise(sAmp, nAmp)
    plotSignalAndNoise(subfigs[0], time, signal, sAmp, noise, nAmp, "Equal Signal- and Noise Amplitude\n Signal is somewhat visible")
    signals.append(signal)
    noises.append(noise)
    
    sAmp, nAmp = 3, 1
    _, signal, noise = generateSignalAndNoise(sAmp, nAmp)
    plotSignalAndNoise(subfigs[1], time, signal, sAmp, noise, nAmp, "Greater Signal Amplitude\nSignal is clearly visible")
    signals.append(signal)
    noises.append(noise)

    sAmp, nAmp = 2.5, 2
    _, signal, noise = generateSignalAndNoise(sAmp, nAmp)
    plotSignalAndNoise(subfigs[2], time, signal, sAmp, noise, nAmp, "Greater Noise Amplitude\nSignal is hard to see")
    signals.append(signal)
    noises.append(noise)

    # plt.show()

    return time, signals, noises


def task2(time, signals, noises):
    """
        Task 2 - Calculating the signal-to-noise ratio (SNR)
        1. Calculate the signal-to-noise ratio from the defined signal scenarios in (3). Use decibel
            scale to report it.
        2. Let's make a target SNR parameter (in dB) to set the signal + noise conditions for further
            analysis.
            a) Start with no scaling of the signal or noise amplitude. i.e. unit signal amplitude,
                and unit noise variance.
            b) Estimate the SNR using the formula in (1) SNR_est
            c) Scale the (unit) noise amplitude with a target SNR (user defined parameter in dB)
                to get the desired SNR for further analysis: scale = sqrt(SNR_est / 10^(SNR/10)).
                a. The sqrt() is to get from power to amplitude when scaling.
                b. The 10^(SNR/10) is because we set the target SNR in dB
            d) Scale the noise signal to get the desired SNR.
            e) Calculate again the SNR_est from (1) after scaling to check that you reach the
                target SNR.
    """

    def plotCombinedSNR(ax, time, signal, noise, snrDb, targetSnrDb=None):
        ax.plot(time, signal + noise)
        if not targetSnrDb:
            ax.set_title(f"Signal + Noise (SNR: {snrDb:.2f} dB)")
        else:
            ax.set_title(f"Signal + Noise (SNR: {snrDb:.2f} dB, Target SNR: {targetSnrDb} dB)")
    
    _, axes = plt.subplots(nrows=1, ncols=3)
    for signal, noise, ax in zip(signals, noises, axes):
        snr = calculateSnrDb(signal, noise)
        plotCombinedSNR(ax, time, signal, noise, snr)

    # plt.show()
    
    signal = signals[0]
    noise = noises[0]

    snrNoScaling = calculateSnr(signal, noise)
    TARGET_SNR_dB = 7
    scale = np.sqrt(snrNoScaling / 10**(TARGET_SNR_dB/10))  # scale = sqrt(SNR_est / 10^(SNR/10))
    noise *= scale
    snrScaledNoise = calculateSnr(signal, noise)

    _, axes = plt.subplots(nrows=1, ncols=1)
    plotCombinedSNR(axes, time, signal, noise, snrScaledNoise, TARGET_SNR_dB)

    # plt.show()

    return snrScaledNoise

    

def task3(time, signal, noise):
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
        N = len(np.arange(-fs/2, fs/2, 1/fs))
        spectrum = np.fft.fft(signal, N)
        spectrum = np.fft.fftshift(spectrum)
        # TODO: Add scaling. Something someting 1/N
        return spectrum
    
    def plotSpectrum(ax, spectrum, title):
        ax.plot(np.arange(-fs/2, fs/2, 1/fs), np.abs(spectrum))
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Amplitude [A]")
        ax.set_title(title)
        ax.set_xlim(-fs/2, fs/2)

    signalSpectrum = calculatePowerSpectrum(signal)
    noisySpectrum = calculatePowerSpectrum(signal + noise)

    _, axes = plt.subplots(nrows=1, ncols=2)
    plotSpectrum(axes[0], signalSpectrum, "Signal Power Spectrum")
    plotSpectrum(axes[1], noisySpectrum, "Signal + Noise Power Spectrum")

    # plt.show()

def task4():
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
    pass

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
    time, signals, noises = task1()
    scaledNoise = task2(time, signals, noises)
    task3(time, signals[0], scaledNoise)
    task3(time, signals[0], scaledNoise)
    bonusTask()
    plt.show()

if __name__ == '__main__':
    main()
