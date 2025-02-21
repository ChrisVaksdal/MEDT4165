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

        pulsesFig = Figure(1, 2, "Pulses and Spectra", "task1_pulses.png",
                           (60, 30))
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
        pulsesFig.addPlot(0,
                          1,
                          gaussFreqs * 1e-6,
                          gaussSpectrum,
                          xLabel="Frequency [MHz]",
                          yLabel="Power [dB]",
                          xLim=(-15, 15),
                          grid=True)
        pulsesFig.addPlot(0, 1, squareFreqs * 1e-6, squareSpectrum)
        pulsesFig.addLegend(0, 1, ["Gauss Pulse", "Square Pulse"])
        pulsesFig.savePlot()

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

    fastGauss = generateGaussPulse(timeVec, 4 * 1e6, 0.4)
    fastSquare = generateSquarePulse(timeVec, 4 * 1e6, 1 / 0.4)
    fgSpectrum, _ = powerSpectrum(fastGauss, fs)
    fsSpectrum, _ = powerSpectrum(fastSquare, fs)
    fastConvolvedGauss = np.convolve(fastGauss, transducerImpulseResponse,
                                     "same")
    fastConvolvedGauss /= np.max(fastConvolvedGauss)
    fastConvolvedSquare = np.convolve(fastSquare, transducerImpulseResponse,
                                      "same")
    fastConvolvedSquare /= np.max(fastConvolvedSquare)

    fcgSpectrum = transducerFrequencyResponse * fgSpectrum
    fcsSpectrum = transducerFrequencyResponse * fsSpectrum

    if plot:
        convolvedSignalFig = Figure(len(signals), 2, "Convolved Signals",
                                    "task1_convolved.png", (60, 60))
        for i, rSig, cSig, name in zip(range(len(signals)), signals,
                                       combinedSignals, names):
            convolvedSignalFig.addPlot(
                i,
                0,
                timeVec * 1e6,
                rSig,
                title=f"Convolving {name} with transducer",
                xLabel="Time [us]",
                yLabel="Amplitude [A]",
                xLim=(-2.5, 2.5),
                # xticks=np.linspace(timeVec[0] * 1e6, timeVec[-1] * 1e6, 11),
                grid=True)
            convolvedSignalFig.addPlot(i, 0, timeVec * 1e6, cSig)
            convolvedSignalFig.addLegend(
                i, 0, [f"Raw {name}", f"{name} convolved with transducer"])
        minFreq, maxFreq = -f0 * 6 * 1e-6, f0 * 6 * 1e-6
        for i, rSpectrum, cSpectrum, name in zip(range(len(combinedSpectra)),
                                                 spectra, combinedSpectra,
                                                 names):
            convolvedSignalFig.addPlot(
                i,
                1,
                freqs * 1e-6,
                rSpectrum / np.max(rSpectrum),
                title=f"Spectrum of {name} convolved with transducer",
                xLabel="Frequency [MHz]",
                yLabel="Power [dB]",
                xLim=[minFreq, maxFreq],
                xticks=np.arange(minFreq, maxFreq, 2),
                grid=True)
            convolvedSignalFig.addPlot(i, 1, freqs * 1e-6, cSpectrum)
            convolvedSignalFig.addLegend(
                i, 1, [f"Raw {name}", f"{name} convolved with transducer"])

        convolvedSignalFig.savePlot()

        fastFigure = Figure(1, 1, "Transmitting signals with f0=4MHz",
                            "task1_fast_signal.png", (30, 30))
        fastFigure.addSinglePlot(timeVec * 1e6,
                                 fastGauss,
                                 xLabel="Time [us]",
                                 yLabel="Amplitude [A]",
                                 xLim=(-2, 2),
                                 grid=True)
        fastFigure.addSinglePlot(timeVec * 1e6, fastSquare)
        fastFigure.addSinglePlot(timeVec * 1e6, fastConvolvedGauss)
        fastFigure.addSinglePlot(timeVec * 1e6, fastConvolvedSquare)
        fastFigure.addLegend(
            0, 0, ["Gauss", "Square", "Gauss Convolved", "Square Convolved"])
        fastFigure.savePlot()

        fSpectrumFig = Figure(1, 1,
                              "Spectra of transmitting signals with f0=4MHz",
                              "task1_fast_spectrum.png", (30, 30))
        fSpectrumFig.addSinglePlot(freqs * 1e-6,
                                   fgSpectrum / np.max(fgSpectrum),
                                   xLabel="Frequency [MHz]",
                                   yLabel="Power [dB]",
                                   xLim=[minFreq, maxFreq],
                                   xticks=np.arange(minFreq, maxFreq, 2),
                                   grid=True)
        fSpectrumFig.addSinglePlot(freqs * 1e-6,
                                   fsSpectrum / np.max(fsSpectrum))
        fSpectrumFig.addSinglePlot(freqs * 1e-6,
                                   fcgSpectrum / np.max(fcgSpectrum))
        fSpectrumFig.addSinglePlot(freqs * 1e-6,
                                   fcsSpectrum / np.max(fcsSpectrum))
        fSpectrumFig.addLegend(
            0, 0, ["Gauss", "Square", "Gauss Convolved", "Square Convolved"])
        fSpectrumFig.savePlot()

    return combinedSignals, combinedSpectra


# def task1_different_frequency(timeVec,
#                               freqs,
#                               originalSignals,
#                               originalSpectra,
#                               fasterSignals,
#                               fasterSpectra,
#                               plot=False):
#     ...
#     if plot:
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(60, 40))
#         for sig, fSig in zip(originalSignals, fasterSignals):
#             ...


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

    # fastGPulse, fastSPulse, _, _, _ = task1_transmission(timeVec, 4 * 1e6)
    # fastConvolvedSignals, fastConvolvedSpectra = task1_combined(
    #     timeVec,
    #     f0,
    #     transducerImpulse,
    #     fftFreqs,
    #     transducerFrequency, [fastGPulse, fastSPulse], [fastGPulse, fastSPulse],
    #     ["Higher freq Gauss Pulse", "Higher freq Square Pulse"],
    #     plot=False)

    # task1_different_frequency(timeVec, fftFreqs, convolvedSignals,
    #                           convolvedSpectra, fastConvolvedSignals,
    #                           fastConvolvedSpectra)

    # plt.show()

    return timeVec, convolvedSignals


# def pulseEchoResponse(r, t):
#     # G(r, t) = (dd(t - r/c)) / (4 * pi * r))
#     # c = 1540m/s
#     # numerator represents an impulse at time t=r/c, when scatterer is met
#     # denominator represents amplitude dimishing with distance
#     # dd is the dirac delta function
#     # To model two-way response: square signal
#     c = 1540
#     # numerator = np.
#     # denominator = 4 * np.pi * r
#     # return (numerator / denominator)**2

# # def distanceToNearestScatterer(scattererPositions, t):
# #     pos = (t + T / 2) * 1540
# #     nearest = scattererPositions[np.abs(scattererPositions - pos).argmin()]
# #     print(f"t: {t}, pos: {pos}, nearest: {nearest}")
# #     return nearest

# def distanceToNearestScatterer(scattererPositions, r):
#     nearest = scattererPositions[np.abs(scattererPositions - r).argmin()]
#     return nearest

# def depthAxis(scattererPositions: np.ndarray):
#     depth = np.linspace(0, 15 * 1e-2, 1000)
#     sig = np.zeros(len(depth))

#     for r in scattererPositions:
#         sig += signal.unit_impulse(r, len(depth)) * pulseEchoResponse(r, r / 1540)
#         # signal += np.array(
#         #     [pulseEchoResponse(np.abs(d - r), d / 1540) for d in depth])
#     sig /= np.max(signal)

#     # scatterSignal = np.array([
#     #     pulseEchoResponse(distanceToNearestScatterer(scattererPositions, r),
#     #                       r / 1540) for r in depth
#     # ])

#     return depth, sig
#     # depth = np.array([pulseEchoResponse(distanceToNearestScatterer(scattererPositions, t), t) for t in timeVec])
#     # for r, t in zip(scattererTimes, scattererPositions):
#     #     closestTime = np.argmin(np.abs(timeVec - t))
#     #     width = 10
#     #     for tt in np.arange(closestTime - width, closestTime + width):
#     #         depth[tt] = pulseEchoResponse(tt/1540, tt)
#     # return depth

# def task2(timeVec, transmittedPulses, plot=False):
#     scattererPositionsCm = np.array([1, 3, 5, 7, 9, 11, 13])
#     depth, signal = depthAxis(scattererPositionsCm * 1e-2)
#     depthFig = Figure(1, 1, "Depth Response", "task2_depth.png", (30, 30))
#     depthFig.addPlot(0, 0, depth * 1e2, signal)
#     plt.show()


def main():
    print("Exercise 2")

    timeVec, pulses = task1(plot=True)
    # task2(timeVec, pulses, plot=True)


if __name__ == '__main__':
    main()
