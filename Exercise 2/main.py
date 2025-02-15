from typing import Tuple
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# def plot(axes, x, y, title:str|None=None, xLabel:str|None=None, yLabel:str|None=None, xLim:Tuple|None=None, yLim:Tuple|None=None, xticks:list[int]|None=None, yticks:list[int]|None=None):


def task1():
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
    pulseTimeVec = np.linspace(0, 1 / f0, int(1 / bw))
    squarePulse = signal.square(2 * np.pi * f0 * pulseTimeVec)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(50, 30))

    axes[0].plot(timeVec * 1e6, gausspulse)
    axes[0].plot(timeVec * 1e6, iGausspulse)
    axes[0].plot(timeVec * 1e6, envGaussPulse)
    axes[0].legend(["Real part", "Imaginary Part", "Envelope"])
    axes[0].set_title(f"Gauss Pulse (f0={f0*1e-6} MHz, bw={bw_abs*1e-6} MHz)")

    padWidth = int((len(timeVec) - len(squarePulse)) / 2)
    paddedSquarePulse = np.pad(squarePulse, (padWidth, padWidth + 1),
                               "constant",
                               constant_values=0)
    axes[1].plot(timeVec * 1e6, paddedSquarePulse)
    axes[1].set_title(f"Square Pulse (f0={f0*1e-6} MHz, T={1/f0*1e6:.2f} us)")

    # Spectra

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 20))

    gaussSpectrum = np.fft.fft(gausspulse)
    gaussSpectrum = (1 / len(gausspulse)) * np.abs(gaussSpectrum)**2
    # gaussSpectrum = np.fft.fftshift(gaussSpectrum)
    gaussSpectrum = 20 * np.log10(gaussSpectrum)

    squareSpectrum = np.fft.fft(paddedSquarePulse)
    squareSpectrum = np.fft.fftshift(squareSpectrum)
    squareSpectrum = 20 * np.log10(np.abs(squareSpectrum))

    freqs = np.fft.fftfreq(len(timeVec), d=1 / fs)
    # freqs = np.fft.fftshift(freqs)

    axes[2].plot(freqs * 1e-6, 20 * np.log10(np.abs(gaussSpectrum)))
    # axes[2].plot(freqs*1e-6, 20 * np.log10(np.abs(squareSpectrum)))

    # axes[2].set_xlim(-(f0*5*1e-6), (f0*5*1e-6))
    # axes[2].set_ylim(-45, 5)

    axes[2].set_title(f"Power Spectra (f0={f0*1e-6} MHz, T={1/f0*1e6:.2f} us)")
    axes[2].legend(["Gauss Pulse", "Square Pulse"])

    plt.show()


def main():
    print("Exercise 2")
    task1()


if __name__ == '__main__':
    main()
