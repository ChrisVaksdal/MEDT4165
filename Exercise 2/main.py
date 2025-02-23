import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .Figure import Figure


def powerSpectrum(signal, fs):
    N = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal, N))
    spectrum = (1 / N) * np.abs(spectrum)**2
    spectrum = 10 * np.log10(spectrum + np.finfo(float).eps)
    freqs = np.fft.fftfreq(N, d=1 / fs)
    freqs = np.fft.fftshift(freqs)
    return spectrum, freqs


def zeroPad(signal, N):
    padWidth = int((N - len(signal)) / 2)
    if len(signal) % 2 == 0:
        return np.pad(signal, padWidth)
    return np.pad(signal, (padWidth, padWidth + 1))


def generateGaussPulse(timeVec, fc, bw):
    return signal.gausspulse(timeVec, fc=fc, bw=bw)


def generateSquarePulse(timeVec, fc, nCycles):
    period = 1 / fc
    pulseTime = period * nCycles
    pulseTimeVec = np.arange(0, pulseTime, 1 / fs)
    pulse = signal.square(2 * np.pi * fc * pulseTimeVec)
    return zeroPad(pulse, len(timeVec))


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
    signalEnv = abs(signal.hilbert(gausspulse))
    threshold = max(signalEnv) / 2

    aboveHalfMax = np.where(signalEnv > threshold)[0]
    startIdx = aboveHalfMax[0]
    endIdx = aboveHalfMax[-1]
    pulseLength = (timeVec[endIdx] - timeVec[startIdx]) / 2
    pulseLengthMillimeters = pulseLength * 1540 * 1e3

    print(f"Pulse length: {pulseLengthMillimeters:.2f} mm")

    if plot:
        gaussFig = Figure(1, 1, "Gaussian Weighted Sinusoidal",
                          "task1_gauss_pulse.png")
        gaussFig.addPlot(0,
                         0,
                         timeVec * 1e6,
                         gausspulse,
                         xLabel="Time [us]",
                         yLabel="Amplitude [A]",
                         grid=True)
        gaussFig.addSinglePlot(timeVec * 1e6, iGausspulse, grid=True)
        gaussFig.addSinglePlot(timeVec * 1e6, envGaussPulse)
        gaussFig.addLegend(0, 0, ["Real part", "Imaginary Part", "Envelope"])
        gaussFig.savePlot()

        pulsesFig = Figure(2, 1, "Pulses and Spectra", "task1_pulses.png",
                           (60, 40))
        pulsesFig.addPlot(0,
                          0,
                          timeVec * 1e6,
                          gausspulse,
                          xLabel="Time [us]",
                          yLabel="Amplitude [A]",
                          xLim=(-2, 2),
                          grid=True)
        pulsesFig.addPlot(0, 0, timeVec * 1e6, paddedSquarePulse)
        pulsesFig.addLegend(0, 0, ["Gauss Pulse", "Square Pulse"])
        pulsesFig.addPlot(1,
                          0,
                          gaussFreqs * 1e-6,
                          gaussSpectrum,
                          xLabel="Frequency [MHz]",
                          yLabel="Power [dB]",
                          xLim=(-15, 15),
                          yLim=(-30, 15),
                          grid=True)
        pulsesFig.addPlot(1, 0, squareFreqs * 1e-6, squareSpectrum)
        pulsesFig.addLegend(1, 0, ["Gauss Pulse", "Square Pulse"])
        pulsesFig.savePlot()

        pulseLengthFig = Figure(1, 1, "Pulse Length Estimation",
                                "task1_pulse_length.png")
        pulseLengthFig.addPlot(0,
                               0,
                               timeVec * 1e6,
                               gausspulse,
                               xLabel="Time [us]",
                               yLabel="Amplitude [A]",
                               xLim=(-2, 2),
                               grid=True,
                               dotted=True)
        pulseLengthFig.addPlot(0, 0, timeVec * 1e6, signalEnv)
        pulseLengthFig.addPlot(0,
                               0,
                               timeVec[aboveHalfMax] * 1e6,
                               signalEnv[aboveHalfMax],
                               emphasized=True)
        pulseLengthFig.addLegend(
            0, 0, ["Gauss Pulse", "Signal Envelope", "Values above half max"])
        pulseLengthFig.savePlot()

    return gausspulse, paddedSquarePulse, gaussFreqs, gaussSpectrum, squareSpectrum


def task1_transducer(timeVec,
                     fc,
                     signals: list,
                     freqs,
                     spectra: list,
                     names: list,
                     plot=False):
    bw = 0.4

    impulseResponse = signal.gausspulse(timeVec, fc=fc, bw=bw)
    impulseResponse = impulseResponse / max(impulseResponse)
    freqResponse, transducerFreqs = powerSpectrum(impulseResponse, fs)

    if plot:
        transducerFig = Figure(2, 1, "Transducer Response",
                               "task1_transducer.png", (30, 30))

        transducerFig.addPlot(0,
                              0,
                              timeVec * 1e6,
                              impulseResponse,
                              title=f"Impulse Response",
                              xLabel="Time [us]",
                              yLabel="Amplitude [A]",
                              xLim=[-T * 1e6 / 4, T * 1e6 / 4],
                              grid=True,
                              emphasized=True)
        for sig in signals:
            transducerFig.addPlot(0, 0, timeVec * 1e6, sig, dotted=True)
        transducerFig.addLegend(0, 0, ["Transducer"] + names)

        transducerFig.addPlot(1,
                              0,
                              transducerFreqs * 1e-6,
                              freqResponse,
                              title=f"Frequency Response",
                              xLabel="Frequency [MHz]",
                              yLabel="Power [dB]",
                              xLim=[-(fc * 6 * 1e-6), (fc * 6 * 1e-6)],
                              grid=True,
                              emphasized=True)
        for spectrum in spectra:
            transducerFig.addPlot(1, 0, freqs * 1e-6, spectrum, dotted=True)
        transducerFig.addLegend(1, 0, ["Transducer"] + names)
        transducerFig.savePlot()

    return impulseResponse, freqResponse


def plotSignalsAndSpectra(title, timeVec, signals, freqs, spectra, names):
    filename = f"task1_{title.lower().replace(' ', '_')}.png"
    signalsFig = Figure(2, 1, title, filename, (30, 30))
    signalsFig.addPlot(0,
                       0,
                       timeVec * 1e6,
                       signals[0],
                       xLabel="Time [us]",
                       yLabel="Amplitude [A]",
                       xLim=(-2.5, 2.5),
                       grid=True,
                       dotted=True)
    signalsFig.addPlot(0, 0, timeVec * 1e6, signals[1], emphasized=True)
    signalsFig.addLegend(0, 0, names)
    signalsFig.addPlot(1,
                       0,
                       freqs * 1e-6,
                       spectra[0],
                       xLabel="Frequency [MHz]",
                       yLabel="Power [dB]",
                       xLim=[-(f0 * 6 * 1e-6), (f0 * 6 * 1e-6)],
                       grid=True,
                       dotted=True)
    signalsFig.addPlot(1, 0, freqs * 1e-6, spectra[1], emphasized=True)
    signalsFig.addLegend(1, 0, names)
    signalsFig.savePlot()


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

    fastGauss = generateGaussPulse(timeVec, 4 * 1e6, 0.4)
    fastSquare = generateSquarePulse(timeVec, 4 * 1e6, 1 / 0.4)
    signals = signals + [fastGauss, fastSquare]
    spectra = spectra + [
        powerSpectrum(fastGauss, fs)[0],
        powerSpectrum(fastSquare, fs)[0]
    ]

    for sig in signals:
        convolvedSignal = np.convolve(sig, transducerImpulseResponse, "same")
        convolvedSignal /= np.max(convolvedSignal)
        combinedSignals.append(convolvedSignal)

        convolvedSpectrum, _ = powerSpectrum(convolvedSignal, fs)
        combinedSpectra.append(convolvedSpectrum)

    if plot:
        plotSigs = list(zip(signals, combinedSignals))
        plotSpectra = list(zip(spectra, combinedSpectra))
        plotSignalsAndSpectra("Gauss Signals", timeVec, plotSigs[0], freqs,
                              plotSpectra[0], ["Gauss", "Convolved Gauss"])
        plotSignalsAndSpectra("Square Signals", timeVec, plotSigs[1], freqs,
                              plotSpectra[1], ["Square", "Convolved Square"])
        plotSignalsAndSpectra(
            "High Frequency Gauss Signals", timeVec, plotSigs[2], freqs,
            plotSpectra[2],
            ["High Frequency Gauss", "Convolved High Frequency Gauss"])
        plotSignalsAndSpectra(
            "High Frequency Square Signals", timeVec, plotSigs[3], freqs,
            plotSpectra[3],
            ["High Frequency Square", "Convolved High Frequency Square"])

    return combinedSignals, combinedSpectra


def task1(plot=False):
    timeVec = np.linspace(-T / 2, T / 2, int(T * fs))

    gPulse, sPulse, fftFreqs, gSpectrum, sSpectrum = task1_transmission(
        timeVec, f0, plot=plot)

    transducerImpulse, transducerFrequency = task1_transducer(
        timeVec,
        f0, [gPulse, sPulse],
        fftFreqs, [gSpectrum, sSpectrum], ["Gauss Pulse", "Square Pulse"],
        plot=plot)

    convolvedSignals, convolvedSpectra = task1_combined(
        timeVec,
        f0,
        transducerImpulse,
        fftFreqs,
        transducerFrequency, [gPulse, sPulse], [gSpectrum, sSpectrum],
        ["Gauss Pulse", "Square Pulse"],
        plot=plot)

    return timeVec, convolvedSignals


def getNoiseTargetSNR(signal, targetSNRdB):
    noise = np.random.randn(len(signal))
    power = 1 / np.linalg.norm(noise) * np.linalg.norm(signal)
    return noise * power / (10**(targetSNRdB / 20))


def task2_receive(pulses, plot=False):
    timeVec = np.arange(-5 / f0, 2 * 15e-2 / 1540, 1 / fs)
    scattererPositionsCm = np.array([1, 3, 5, 7, 9, 11, 13])
    scatterIndices = [
        np.argmin(np.abs(timeVec - 2 * r / 1540))
        for r in scattererPositionsCm * 1e-2
    ]

    gamma = 1e-2  # Picked at random
    reflections = np.zeros_like(timeVec)
    reflections[scatterIndices] = gamma / (4 * np.pi * scattererPositionsCm *
                                           1e-2)

    gauss = pulses[0][700:1800]  # Slice to avoid spending forever on zeros
    square = pulses[1][700:1800]

    receivedGauss = np.convolve(gauss, reflections, "same")
    receivedSquare = np.convolve(square, reflections, "same")

    SNR = 20
    noisyGauss = receivedGauss + getNoiseTargetSNR(receivedGauss, SNR)
    noisySquare = receivedSquare + getNoiseTargetSNR(receivedSquare, SNR)

    if plot:
        depthFig = Figure(1, 1, "Received signals without noise",
                          "task2_depth_raw.png", (60, 30))
        depthFig.addPlot(0,
                         0,
                         timeVec * 1e2 * 1540,
                         receivedGauss,
                         xLabel="Depth (cm)",
                         yLabel="Amplitude",
                         grid=True)
        depthFig.addPlot(0, 0, timeVec * 1e2 * 1540, receivedSquare)
        depthFig.addLegend(0, 0, ["Gauss", "Square"])
        depthFig.savePlot()

        noisyDepthFig = Figure(1, 1, "Received signals with noise",
                               "task2_depth_noisy.png", (60, 30))
        noisyDepthFig.addPlot(0,
                              0,
                              timeVec * 1e2 * 1540,
                              noisyGauss,
                              xLabel="Depth (cm)",
                              yLabel="Amplitude",
                              grid=True)
        noisyDepthFig.addPlot(0, 0, timeVec * 1e2 * 1540, noisySquare)
        noisyDepthFig.addLegend(0, 0, ["Gauss", "Square"])
        noisyDepthFig.savePlot()

    return noisyGauss, noisySquare


def task2_filter(pulses, plot=False):
    fHigh = 3 * 1e6
    fLow = 2 * 1e6
    [b, a] = signal.butter(5, [fLow, fHigh], btype="bandpass", fs=fs)

    gaussSpectrum, freqs = powerSpectrum(pulses[0], fs)
    squareSpectrum, _ = powerSpectrum(pulses[1], fs)
    spectra = [gaussSpectrum, squareSpectrum]

    [w, p] = signal.freqz(b, a, len(freqs), whole=True, fs=fs)
    p = 20 * np.log10(np.fft.fftshift(abs(p) + np.finfo(float).eps))

    if plot:
        filterFigure = Figure(1, 1, "Bandpass filter", "task2_filter.png",
                              (60, 30))
        filterFigure.addPlot(0,
                             0,
                             w,
                             p,
                             xLabel="Frequency (Hz)",
                             yLabel="Amplitude",
                             grid=True)
        for i, spectrum in enumerate(spectra):
            filterFigure.addPlot(0,
                                 0,
                                 freqs * 1e-6,
                                 spectrum,
                                 xLabel="Frequency (Hz)",
                                 yLabel="Amplitude",
                                 xLim=[-10, 10],
                                 grid=True)


def task2(pulseTimeVec, pulses, plot=False):
    receivedGauss, receivedSquare = task2_receive(pulses, plot=plot)
    task2_filter([receivedGauss, receivedSquare], plot=plot)


def main():
    print("Exercise 2")

    timeVec, pulses = task1(plot=True)
    task2(timeVec, pulses, plot=True)
    # plt.show()


if __name__ == '__main__':
    main()
