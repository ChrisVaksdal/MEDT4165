import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from .Figure import Figure


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

        gaussPulseFig = Figure(1, 2, "Gauss Pulse", "task1_gauss.png")
        gaussPulseFig.addPlot(0,
                              0,
                              timeVec * 1e6,
                              gausspulse,
                              title=f"Signal",
                              xLabel="Time [us]",
                              yLabel="Amplitude [A]",
                              grid=True)
        gaussPulseFig.addPlot(
            0,
            1,
            gaussFreqs * 1e-6,
            gaussSpectrum,
            title=f"Power Spectrum",
            xLabel="Frequency [MHz]",
            yLabel="Power [dB]",
            xLim=[-(f0 * 5 * 1e-6), (f0 * 5 * 1e-6)],
            # yLim=[-45, 10],
            grid=True)
        gaussPulseFig.savePlot()

        squarePulseFig = Figure(1, 2, "Square Pulse", "task1_square.png")
        squarePulseFig.addPlot(0,
                               0,
                               timeVec * 1e6,
                               paddedSquarePulse,
                               title=f"Signal",
                               xLabel="Time [us]",
                               yLabel="Amplitude [A]",
                               grid=True)
        squarePulseFig.addPlot(
            0,
            1,
            squareFreqs * 1e-6,
            squareSpectrum,
            title=f"Power Spectrum (f0={f0*1e-6} MHz, T={1/f0*1e6:.2f} us)",
            xLabel="Frequency [MHz]",
            yLabel="Power [dB]",
            xLim=[-(f0 * 5 * 1e-6), (f0 * 5 * 1e-6)],
            # yLim=[-45, 10],
            grid=True)
        squarePulseFig.savePlot()

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
                              grid=True)
        for sig in signals:
            transducerFig.addPlot(
                0,
                0,
                timeVec * 1e6,
                sig,
            )
        transducerFig.addLegend(0, 0, ["Transducer"] + names)

        transducerFig.addPlot(1,
                              0,
                              transducerFreqs * 1e-6,
                              freqResponse,
                              title=f"Frequency Response",
                              xLabel="Frequency [MHz]",
                              yLabel="Power [dB]",
                              xLim=[-(fc * 6 * 1e-6), (fc * 6 * 1e-6)],
                              grid=True)
        for spectrum in spectra:
            transducerFig.addPlot(
                1,
                0,
                freqs * 1e-6,
                spectrum,
            )
        transducerFig.addLegend(1, 0, ["Transducer"] + names)
        transducerFig.savePlot()

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
        convolvedSignalFig = Figure(len(signals), 2, "Convolved Signals",
                                    "task1_convolved.png", (60, 60))
        for i, rSig, cSig, name in zip(range(len(signals)), signals,
                                       combinedSignals, names):
            convolvedSignalFig.addPlot(i,
                                       0,
                                       timeVec * 1e6,
                                       rSig,
                                       title=f"Convolving {name} with transducer",
                                       xLabel="Time [us]",
                                       yLabel="Amplitude [A]",
                                       xLim=[-T * 1e6 / 4, T * 1e6 / 4],
                                       xticks=np.linspace(timeVec[0] * 1e6, timeVec[-1] * 1e6, 11),
                                       grid=True)
            convolvedSignalFig.addPlot(i,
                                       0,
                                       timeVec * 1e6,
                                       cSig)
            convolvedSignalFig.addLegend(i, 0, [f"Raw {name}", f"{name} convolved with transducer"])
        minFreq, maxFreq = -f0 * 6 * 1e-6, f0 * 6 * 1e-6
        for i, rSpectrum, cSpectrum, name in zip(range(len(combinedSpectra)),
                                                 spectra, combinedSignals,
                                                 names):
            convolvedSignalFig.addPlot(i,
                                       1,
                                       freqs * 1e-6,
                                       rSpectrum / np.max(rSpectrum),
                                       title=f"Spectrum of {name} convolved with transducer",
                                       xLabel="Frequency [MHz]",
                                       yLabel="Power [dB]",
                                       xLim=[minFreq, maxFreq],
                                       xticks=np.arange(minFreq, maxFreq, 2),
                                       grid=True)
            convolvedSignalFig.addPlot(i,
                                       1,
                                       freqs * 1e-6,
                                       cSpectrum)
            convolvedSignalFig.addLegend(i, 1, [f"Raw {name}", f"{name} convolved with transducer"])

        convolvedSignalFig.savePlot()

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

    transducerImpulse, transducerFrequency = task1_transducer(
        f0,
        timeVec, [gPulse, sPulse],
        fftFreqs, [gSpectrum, sSpectrum], ["Gauss Pulse", "Square Pulse"],
        plot=True)

    convolvedSignals, convolvedSpectra = task1_combined(
        timeVec,
        f0,
        transducerImpulse,
        fftFreqs,
        transducerFrequency, [gPulse, sPulse], [gSpectrum, sSpectrum],
        ["Gauss Pulse", "Square Pulse"],
        plot=True)

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
