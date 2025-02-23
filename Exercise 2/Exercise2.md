# MEDT4165 - Exercise 2 Ultrasound Transmission and Reception

Written by: Christoffer-Robin Vaksdal

## Part 1 - Transmission

### Generating signals

For this exercise we are starting out with generating two signals, a
Gaussian-weighted sinusoidal pulse and a square wave pulse. Both signals have
a frequency `f0 = 2.5MHz`. We can define these signals as `numpy.ndarray`s
using `scipy`, starting with the gaussian pulse:

```python
fs = 250 * 1e6      # Sample rate
f0 = 2.5 * 1e6      # Signal base frequency
T = 10 * 1e-6       # Length of time vector
N = int(T * fs)     # Number of samples

bw = 0.3 # Relative bandwidth. bw_abs = 7.5MHz

timeVec = np.linspace(-T / 2, T / 2, N)

gaussPulse, _imagGaussPulse, _envGaussPulse = signal.gausspulse(timeVec,
                                                               fc=f0,
                                                               bw=bw,
                                                               retenv=True,
                                                               retquad=True)
```

So we define a time-vector, establish a sampling frequency and base frequency
of the signal, as well as a relative bandwidth. The arguments `retenv` and
`retquad` are for returning the signal envelope and imaginary part,
respectively. Here's a plot of all three:

>> ![task1_gauss_pulse](figures/task1_gauss_pulse.png)
>> Plot of gaussian pulse. Along with the signal you can see the imaginary
>> parts and the envelope of the signal.

Now we can quickly calculate the length of the gauss-pulse and generate a
square-wave pulse with roughly the same length.

```python
period = 1 / f0
nPeriods = int(1 / bw)
pulseTime = period * nPeriods
pulseTimeVec = np.arange(0, pulseTime, 1 / fs)
squarePulse = signal.square(2 * np.pi * f0 * pulseTimeVec)
```

This pulse will be shorter than the full signal vector of the gauss pulse, so
we can define a function to pad the signal with zeros:

```python
def zeroPad(signal, N):
    padWidth = int((N - len(signal)) / 2)
    if len(signal) % 2 == 0:
        return np.pad(signal, padWidth)
    return np.pad(signal, (padWidth, padWidth + 1))

paddedSquarePulse = zeroPad(squarePulse, len(gaussPulse))
```

Here are the pulses and their respective spectra:

>> ![task1_pulses](figures/task1_pulses.png)
>> Gauss- and square pulse with associated power spectra.

From the signal spectra we observe that the gaussian weighted sinusoidals only
have peaks around the base frequency `f0 = 2.5MHz`. The square pulse on the
other hand, has a number of harmonics which we can see as the smaller peaks.

As expected, the pulses have roughly the same length in time and from the
spectra we can also see that both signals have a significant peak around
`f = f0 = 2.5MHz`. Note that the gaussian pulse has *all* its energy spread at
`&plusmn;2.5MHz`, while the square pulse also has smaller peaks at odd-numbered
multiples of the base frequency (harmonics!).

### Calculating Spatial Pulse Length (SPL)

To calculate the spatial length of the pulses we use the Hilbert function to
find the pulse envelope, then find the 'ends' of the pulse as the points on
either end where the amplitude of the signal has dropped to half of the maximum
amplitude. This duration can be converted to a spatial length using the speed
of sound `c = 1540m/s`.

```python
signalEnv = abs(signal.hilbert(gausspulse))
threshold = max(signalEnv) / 2

aboveHalfMax = np.where(signalEnv > threshold)[0]
startIdx = aboveHalfMax[0]
endIdx = aboveHalfMax[-1]
pulseLength = (timeVec[endIdx] - timeVec[startIdx]) / 2
pulseLengthMillimeters = pulseLength * 1540 * 1e3
```

Performing this calculation we get an output of a pulse duration of 511 samples,
which comes out to a spatial pulse length of 0.90mm. The signal envelope with
the relevant section emphasized can be seen below:

>> ![task1_pulse_length](figures/task1_pulse_length.png)
>> Plot of gaussian pulse with area where envelope is above half of max
>> emphasized.

### Transducer Response

To make the transducer response as a gaussian-weighted sinusodial we can reuse
what we learned in the previous section:

```python
impulseResponse = signal.gausspulse(timeVec, fc=2.5*1e6, bw=0.4)
freqResponse, transducerFreqs = powerSpectrum(impulseResponse, fs)
```

Here's what the impulse- and frequency responses look like along with the
pulses from before to compare:

>> ![task1_transducer](figures/task1_transducer.png)
>> Transducer responses along with signals and spectra.

### Transducer Influence on Signal

Now to see what happens when the signal transducer interact with our pulses.
The way these signals combine is called convolution in the time domain, and is
equivalent to multiplication in the frequency domain. Here's some python code
for combining the signal with the transducer response:

```python
convolvedSignal = np.convolve(sig, impulseResponse, "same")
convolvedSignal /= np.max(convolvedSignal)  # Normalize signal

convolvedSpectrum = powerSpectrum(convolvedSignal, fs)
```

We use `np.convolve()` to perform the convolution, specifying `"same"` makes
the output signal be the same length (same number of samples) as the input (as
opposed to twice as long). In the following figures you can see the convolved
signals and associated power spectra.

>> ![task1_gauss_signals](figures/task1_gauss_signals.png)
>> Gauss pulse and powerspectrum along with transducer responses and convolved
>> results.
-----
>> ![task1_square_signals](figures/task1_square_signals.png)
>> Square pulse and powerspectrum along with transducer responses and convolved
>> results.

In the convolved plots, note that the gauss pulse is largely unchanged in terms
of its shape. This should not be surprising, they are almost the same signal
after all. The square pulse loses its square shape, becoming rounded by the
transducer. This effect is apparant in the spectrum of the square signal as
well, where you can see the harmonics have been eliminated.

If we up the transmit frequency to 4MHz, but keep the transducer the same, we
get signals and spectra like these:

>> ![task1_high_frequency_gauss_signals](figures/task1_high_frequency_gauss_signals.png)
>> Higher frequency Gauss pulse convolved with transducer.
-----
>> ![task1_high_frequency_square_signals](figures/task1_high_frequency_square_signals.png)
>> Higher frequency square pulse convolved with transducer.

We see that when convolving the higher frequency transmitted signal with the
transducer can mess up the signal. The shape of the signal is altered and we
see in the spectra that the frequency peaks have moved to somewhere between the
original transmitted frequency and the center frequency of the transducer.

I believe the pulse energy outside of the transducer bandwidth is lost to heat
generated through friction and other mechanical or electromagnetic resistances
in the transducer. In other words, that energy will go towards forcing the
transducer to move in a way it would not otherwise move and that resistance to
moving in that way would generate heat.

## Part 2 - Reception

### The Received Signal

The received signal is a time-variant signal as before, encoding within it
information about distances. To generate this received signal over time, we are
going to imagine a sound wave travelling in a line that's 15 cm long. Along the
way, at 1, 3, 5, 7, 9, 11 and 13 cm distance, there are scatterers that will
produce reflections/echoes as the wave travels through. To create the reflected
signal we receive, we define a depth axis containing the scatterer impulses
(pulse echo response means squared Green's function. Impulses are scaled with
depth) and use that as a propagation filter on the transmitted signal from
before.

```python
timeVec = np.arange(-5 / f0, 2 * 15e-2 / 1540, 1 / fs)
scattererPositionsCm = np.array([1, 3, 5, 7, 9, 11, 13])
scatterIndices = [
    np.argmin(np.abs(timeVec - 2 * r / 1540))
    for r in scattererPositionsCm * 1e-2
]

gamma = 1e-2    # Picked at random
reflections = np.zeros_like(timeVec)
reflections[scatterIndices] = gamma / (4 * np.pi * scattererPositionsCm *
                                        1e-2)

gauss = pulses[0][700:1800]     # Slice to avoid spending forever on zeros
square = pulses[1][700:1800]

receivedGauss = np.convolve(gauss, reflections, "same")
receivedSquare = np.convolve(square, reflections, "same")
```

Using this code we can plot the received signal as a function of depth:

>> ![task2_depth_raw](figures/task2_depth_raw.png)
>> Received signal over time converted to depth axis.

Here we see the impulses coming from the transmitted signal being reflected by
the reflectors. We can also define a function to generate noise with a target
SNR based on what we learned in the previous exercise:

```python
def getNoiseTargetSNR(signal, targetSNRdB):
    noise = np.random.randn(len(signal))
    power = 1 / np.linalg.norm(noise) * np.linalg.norm(signal)
    return noise * power / (10**(targetSNRdB / 20))

SNR = 20
noisyGauss = receivedGauss + getNoiseTargetSNR(receivedGauss, SNR)
noisySquare = receivedSquare + getNoiseTargetSNR(receivedSquare, SNR)
```

>> ![task2_depth_noisy](figures/task2_depth_noisy.png)
>> Signal depth axis with added noise.

### Filtering the received signal

We can reuse what we learned in exercise 1 to create a bandpass filter to
eliminate noise from our signal:

```python
fHigh = 3 * 1e6
fLow = 2 * 1e6
[b, a] = signal.butter(4, [fLow, fHigh], btype="bandpass", fs=fs)

# Estimate frequency response
[w, p] = signal.freqz(b, a, len(freqs), whole=True, fs=fs)
p = 20 * np.log10(np.fft.fftshift(abs(p) + np.finfo(float).eps))
```

Plotting the frequency response of the filter along with the power spectra of
our signal, we get this:

>> ![task2_filter](figures/task2_filter.png)
>> Frequency response of bandpass-filter along with spectra of received signals.

Now to apply the filter to the received signals we use `scipy.signal.filtfilt`
in order to apply the filter in both directions. this is to preserve phase
information in the signal.

```python
filteredGauss = signal.filtfilt(b, a, noisyGauss)
filteredSquare = signal.filtfilt(b, a, noisySquare)
```

>> ![task2_filtered](figures/task2_filtered.png)
>> Received and filtered signals for gauss pulse and square pulse.

As you can see, the filtered signals preserve the reflected pulses as desired,
but otherwise removes most of the noise present in the signal.

### Time-Gain Compensation (TGC)

We know the received signal is attenuated predictably based on depth. Since
this attenuation only depends on time between sending and receiving (which in
turn is just based on the distance to the scatterer). So we simply define the
inverse of this attenuation as a gain profile we can apply to the received
signal. The resulting signals amplitude is now in theory independant of depth.

```python
rAxis = timeVec * 1540 / 2
depthGain = 20 * (4 * np.pi * rAxis)**2
tgcGauss = filteredGauss * depthGain
tgcSquare = filteredSquare * depthGain
```

The resulting signals (compared to the original, non-compensated signals) look
like this:

>> ![task2_tgc](figures/task2_tgc.png)
>> Received and filtered signals for gauss pulse and square pulse.

In the compensated plots, notice how each of the reflector impulses have the
same amplitude. This has the side effect of also amplifying the noise more and
more, meaning our SNR goes down over time (aka. with increased depth).
