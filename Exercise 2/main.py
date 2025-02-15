from typing import Tuple
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# def plot(axes, x, y, title:str|None=None, xLabel:str|None=None, yLabel:str|None=None, xLim:Tuple|None=None, yLim:Tuple|None=None, xticks:list[int]|None=None, yticks:list[int]|None=None):


def powerSpectrum(signal, fs):
    N = len(signal)
    spectrum = (1 / N) * np.abs(np.fft.fft(signal, N))**2
    # spectrum = 10 * np.log10(spectrum)
    spectrum = np.fft.fftshift(spectrum)
    freqs = np.fft.fftfreq(N, d=1 / fs)
    freqs = np.fft.fftshift(freqs)
    return spectrum, freqs


def zeroPad(signal, N):
    padWidth = int((N - len(signal)) / 2)
    if len(signal) % 2 == 0:
        return np.pad(signal, padWidth)
    return np.pad(signal, (padWidth, padWidth + 1))


def task1_transmission():
    fs = 250 * 1e6
    f0 = 2.5 * 1e6
    T = 10 * 1e-6
    bw = 0.3
    bw_abs = bw * f0

    timeVec = np.linspace(-T / 2, T / 2, int(T * fs))

    # Gauss pulse:
    gausspulse, iGausspulse, envGaussPulse = signal.gausspulse(timeVec,
                                                               fc=f0,
                                                               bw=bw,
                                                               retenv=True,
                                                               retquad=True)

    # Square pulse:
    period = 1 / f0
    nPeriods = int(1 / bw)
    pulseTime = period * nPeriods
    pulseTimeVec = np.arange(0, pulseTime, 1 / fs)

    squarePulse = signal.square(2 * np.pi * f0 * pulseTimeVec)
    paddedSquarePulse = zeroPad(squarePulse, len(timeVec))

    # Spectra
    gaussSpectrum, gaussFreqs = powerSpectrum(gausspulse, fs)
    squareSpectrum, squareFreqs = powerSpectrum(paddedSquarePulse, fs)

    # Pulse length estimation
    signalEnv = abs(signal.hilbert(envGaussPulse))
    threshold = max(signalEnv) / 2

    startIdx = np.where(signalEnv > threshold)[0][0]
    endIdx = np.where(signalEnv > threshold)[0][-1]

    pulseLengthN = endIdx - startIdx
    pulseLength = pulseLengthN / fs

    speedOfSoundMetersPerSecond = 1540
    pulseLengthMillimeters = pulseLength * speedOfSoundMetersPerSecond * 1e3

    print(f"Pulse length: {pulseLengthN} samples")
    print(f"Pulse length: {pulseLengthMillimeters:.2f} mm")

    # Plotting

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))
    axes.plot(timeVec * 1e6, gausspulse)
    axes.plot(timeVec * 1e6, iGausspulse)
    axes.plot(timeVec * 1e6, envGaussPulse)
    axes.plot(timeVec * 1e6, signalEnv)
    axes.legend(["Real part", "Imaginary Part", "Envelope", "Pulse Envelope"])
    axes.set_title(f"Gauss Pulse (f0={f0*1e-6} MHz, bw={bw_abs*1e-6} MHz)")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 40))

    axes[0].plot(timeVec * 1e6, gausspulse)
    axes[0].plot(timeVec * 1e6, paddedSquarePulse)
    axes[0].set_title(f"Gauss Pulse vs. Square pulse")
    axes[0].legend(["Gauss Pulse", "Square Pulse"])

    axes[1].plot(gaussFreqs * 1e-6, gaussSpectrum)
    axes[1].plot(squareFreqs * 1e-6, squareSpectrum)

    axes[1].set_xlim(-(f0 * 5 * 1e-6), (f0 * 5 * 1e-6))
    # axes[1].set_ylim(-45, 10)

    axes[1].set_title(f"Power Spectra (f0={f0*1e-6} MHz, T={1/f0*1e6:.2f} us)")
    axes[1].legend(["Gauss Pulse", "Square Pulse"])

    return timeVec, gausspulse, squarePulse, gaussFreqs, gaussSpectrum, squareSpectrum


def task1_transducer(timeVec, signals: list, freqs, spectra: list):
    fs = 250 * 1e6
    f0 = 2.5 * 1e6
    bw = 0.4
    T = 10 * 1e-6
    timeVec = np.linspace(-T / 2, T / 2, int(T * fs))
    impulseResponse = signal.gausspulse(timeVec, fc=f0, bw=bw)
    freqResponse = signal.hilbert(impulseResponse)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 40))
    axes[0].plot(timeVec * 1e6, impulseResponse)

    return timeVec, impulseResponse, freqResponse


def task1():
    time, gPulse, sPulse, fftFreqs, gSpectrum, sSpectrum = task1_transmission()
    transducerTime, transducerImpulse, transducerFrequency = task1_transducer(
        time, [gPulse, sPulse], fftFreqs, [gSpectrum, sSpectrum])

    plt.show()


def main():
    print("Exercise 2")
    task1()


if __name__ == '__main__':
    main()
