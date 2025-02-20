import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from os import makedirs
from typing import Tuple

from helpers import getOutputDir

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


class Figure:

    def __init__(self,
                 rows: int,
                 cols: int,
                 title: str,
                 filename: str,
                 figsize=(30, 20)):
        self.rows = rows
        self.cols = cols
        self.title = title
        self.filename = filename
        self.fig, self.axes = plt.subplots(nrows=self.rows,
                                           ncols=self.cols,
                                           figsize=figsize)

    def _getAxis(self, row: int, col: int):
        if self.rows == 1 and self.cols == 1:
            return self.axes
        elif self.rows == 1 and self.cols > 1:
            return self.axes[col]
        elif self.cols == 1 and self.rows > 1:
            return self.axes[row]
        else:
            return self.axes[row, col]
    
    def addPlot(self,
                row: int,
                col: int,
                x,
                y,
                title: str | None = None,
                xLabel: str | None = None,
                yLabel: str | None = None,
                xLim: Tuple | None = None,
                yLim: Tuple | None = None,
                xticks: list[int] | None = None,
                yticks: list[int] | None = None,
                grid: bool = False):
        ax = self._getAxis(row, col)
        ax.xaxis.set_tick_params(labelsize=32, width=4)
        ax.yaxis.set_tick_params(labelsize=32, width=4)
        ax.plot(x, y)
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
        if grid:
            ax.grid()
    
    def addSinglePlot(self,
                      x,
                      y,
                      title: str | None = None,
                      xLabel: str | None = None,
                      yLabel: str | None = None,
                      xLim: Tuple | None = None,
                      yLim: Tuple | None = None,
                      xticks: list[int] | None = None,
                      yticks: list[int] | None = None,
                      grid: bool = False):
        self.addPlot(0, 0, x, y, title, xLabel, yLabel, xLim, yLim, xticks, yticks, grid)

    def addLegend(self, row: int, col: int, labels: list[str]):
        ax = self._getAxis(row, col)
        ax.legend(labels, prop={"size": 36})

    def savePlot(self):
        makedirs(f"{getOutputDir(2)}/figures", exist_ok=True)
        self.fig.savefig(f"{getOutputDir(2)}/figures/{self.filename}")


fs = 250 * 1e6
f0 = 2.5 * 1e6
T = 10 * 1e-6


def task1_transmission(timeVec, f0, plot=False):
    bw = 0.3
    bw_abs = bw * f0

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

    if plot:
        gaussFig = Figure(1, 1, "Gauss Weighted Sinusoidal",
                          "task1_gauss_pulse.png")
        gaussFig.addPlot(
            0,
            0,
            timeVec * 1e6,
            gausspulse,
            title=f"Gauss Pulse (f0={f0*1e-6} MHz, bw={bw_abs*1e-6} MHz)",
            xLabel="Time [us]",
            yLabel="Amplitude [A]")
        gaussFig.addSinglePlot(timeVec * 1e6, iGausspulse, grid=True)
        gaussFig.addSinglePlot(timeVec * 1e6, envGaussPulse)
        gaussFig.addLegend(0, 0,["Real part", "Imaginary Part", "Envelope"])

        gaussFig.savePlot()

        signalsFig = Figure(1, 2, "Signals and Spectra", "task1_signals.png")
        signalsFig.addPlot(
            0,
            0,
            timeVec * 1e6,
            gausspulse,
            title=f"Signals (f0={f0*1e-6} MHz, bw={bw_abs*1e-6} MHz)",
            xLabel="Time [us]",
            yLabel="Amplitude [A]",
        )
        signalsFig.addPlot(0, 0, timeVec * 1e6, paddedSquarePulse)
        signalsFig.addLegend(0, 0, ["Gauss Pulse", "Square Pulse"])

        signalsFig.addPlot(
            0,
            1,
            gaussFreqs * 1e-6,
            gaussSpectrum,
            title=f"Power Spectra (f0={f0*1e-6} MHz, T={1/f0*1e6:.2f} us)",
            xLabel="Frequency [MHz]",
            yLabel="Power [dB]",
            xLim=[-(f0 * 5 * 1e-6), (f0 * 5 * 1e-6)],
            # yLim=[-45, 10],
        )

    return gausspulse, paddedSquarePulse, gaussFreqs, gaussSpectrum, squareSpectrum


def task1_transducer(fc,
                     timeVec,
                     signals: list,
                     freqs,
                     spectra: list,
                     names: list,
                     plot=False):
    bw = 0.4

    impulseResponse = signal.gausspulse(timeVec, fc=fc, bw=bw)
    freqResponse, transducerFreqs = powerSpectrum(impulseResponse, fs)

    if plot:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 40))

        axes[0].plot(timeVec * 1e6, impulseResponse)
        for sig in signals:
            axes[0].plot(timeVec * 1e6, sig)
        axes[0].set_title(f"Signals")
        axes[0].legend(["transducer"] + names)
        axes[0].set_xlim(-T * 1e6 / 4, T * 1e6 / 4)

        axes[1].plot(transducerFreqs * 1e-6, freqResponse)
        for spectrum in spectra:
            axes[1].plot(freqs * 1e-6, spectrum)
        axes[1].set_title(f"Frequency Responses")
        axes[1].legend(["transducer"] + names)
        axes[1].set_xlim(-(fc * 6 * 1e-6), (fc * 6 * 1e-6))

    return impulseResponse, freqResponse


def task1_combined(timeVec,
                   f0,
                   transducerImpulseResponse,
                   freqs,
                   transducerFrequencyResponse,
                   signals: list,
                   spectra,
                   names: list,
                   plot=False):

    combinedSignals = []
    combinedSpectra = []

    for sig, spectrum in zip(signals, spectra):
        convolvedSignal = np.convolve(sig, transducerImpulseResponse, "same")
        convolvedSignal /= np.max(convolvedSignal)
        combinedSignals.append(convolvedSignal)

        # spectrum, _ = powerSpectrum(convolvedSignal, fs)
        convolvedSpectrum = transducerFrequencyResponse * spectrum
        combinedSpectra.append(convolvedSpectrum / np.max(convolvedSpectrum))

    if plot:
        fig, axes = plt.subplots(nrows=len(signals), ncols=2, figsize=(60, 40))

        for i, rSig, cSig, name in zip(range(len(signals)), signals,
                                       combinedSignals, names):
            axes[i][0].plot(timeVec * 1e6, rSig)
            axes[i][0].plot(timeVec * 1e6, cSig)
            axes[i][0].set_title(f"Convolving {name} with transducer")
            axes[i][0].legend(
                [f"Raw {name}", f"{name} convolved with transducer"])
            axes[i][0].set_xticks(
                np.linspace(timeVec[0] * 1e6, timeVec[-1] * 1e6, 11))

        for i, rSpectrum, cSpectrum, name in zip(range(len(combinedSpectra)),
                                                 spectra, combinedSignals,
                                                 names):
            axes[i][1].plot(freqs * 1e-6, rSpectrum / np.max(rSpectrum))
            axes[i][1].plot(freqs * 1e-6, cSpectrum)
            axes[i][1].set_title(
                f"Spectrum of {name} with and without transducer influence")
            axes[i][1].legend(
                [f"Raw {name}", f"{name} convolved with transducer"])
            minFreq, maxFreq = -f0 * 6 * 1e-6, f0 * 6 * 1e-6
            axes[i][1].set_xlim((minFreq, maxFreq))
            axes[i][1].set_xticks(np.arange(minFreq, maxFreq, 2))

    return combinedSignals, combinedSpectra


def task1_different_frequency(timeVec, freqs, originalSignals, originalSpectra,
                              fasterSignals, fasterSpectra):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 40))
    for sig, fSig in zip(originalSignals, fasterSignals):
        ...
    ...


def task1():
    timeVec = np.linspace(-T / 2, T / 2, int(T * fs))

    gPulse, sPulse, fftFreqs, gSpectrum, sSpectrum = task1_transmission(
        timeVec, f0, plot=True)

    # transducerImpulse, transducerFrequency = task1_transducer(
    #     f0,
    #     timeVec, [gPulse, sPulse],
    #     fftFreqs, [gSpectrum, sSpectrum], ["Gauss Pulse", "Square Pulse"],
    #     plot=False)

    # convolvedSignals, convolvedSpectra = task1_combined(
    #     timeVec,
    #     f0,
    #     transducerImpulse,
    #     fftFreqs,
    #     transducerFrequency, [gPulse, sPulse], [gSpectrum, sSpectrum],
    #     ["Gauss Pulse", "Square Pulse"],
    #     plot=False)

    # fastGPulse, fastSPulse, _, _, _ = task1_transmission(f0, 4 * 1e6)
    # fastConvolvedSignals, fastConvolvedSpectra = task1_combined(
    #     timeVec,
    #     f0,
    #     transducerImpulse,
    #     fftFreqs,
    #     transducerFrequency, [fastGPulse, fastSPulse],
    #     ["Higher freq Gauss Pulse", "Higher freq Square Pulse"],
    #     plot=False)

    # task1_different_frequency(timeVec, fftFreqs, convolvedSignals,
    #                           convolvedSpectra, fastConvolvedSignals,
    #                           fastConvolvedSpectra)

    # plt.show()


def pulseEchoResponse(r, t):
    # G(r, t) = (sigma * (t - r/c)) / (4 * pi * r))
    # c = 1540m/s
    # numerator represents an impulse at time t=r/c, when scatterer is met
    # denominator represents amplitude dimishing with distance
    # To model two-way response: square signal
    sigma = 1
    c = 1540
    numerator = sigma * (t - r / c)
    denominator = 4 * np.pi * r
    return (numerator / denominator)**2


def depthAxis(timeVec: np.ndarray, scattererPositions: np.ndarray):
    depth = np.zeros((len(timeVec), len(scattererPositions)))
    for i, t in enumerate(timeVec):
        for j, pos in enumerate(scattererPositions):
            depth[i][j] = pulseEchoResponse(pos, t)
    return depth


def task2(timeVec, transmittedPulse):
    scattererPositionsCm = np.array([1, 3, 5, 7, 9, 11, 13])
    depth = depthAxis(timeVec, scattererPositionsCm * 1e-2)


def main():
    print("Exercise 2")
    task1()
    # task2()


if __name__ == '__main__':
    main()
