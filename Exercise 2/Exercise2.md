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

bw = 0.3 # Relative bandwidth. bw_abs = 75MHz

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

Using the power spectrum functionality from Exercise 1 we can plot the signals
and their power spectra:

>> ![task1_gauss](figures/task1_gauss.png)
>> Gaussian pulse with associated power spectrum.
----------
>> ![task1_square](figures/task1_square.png)
>> Gaussian pulse with associated power spectrum.

From the signal spectra we observe that the gaussian weighted sinusoidals only
have peaks around the base frequency `f0 = 2.5MHz`. The square pulse on the
other hand, has a number of harmonics which we can see as the smaller peaks.

### Calculating Spatial Pulse Length (SPL)

To calculate the spatial length of the pulses we use the Hilbert function to
find the pulse envelope, then find the 'ends' of the pulse as the points on
either end where the amplitude of the signal has dropped to half of the maximum
amplitude. This duration can be converted to a spatial length using the speed
of sound `c = 1540m/s`.

```python
signalEnv = abs(signal.hilbert(envGaussPulse))
threshold = max(signalEnv) / 2

startIdx = np.where(signalEnv > threshold)[0][0]
endIdx = np.where(signalEnv > threshold)[0][-1]

pulseLengthN = endIdx - startIdx
pulseLength = pulseLengthN / fs

speedOfSoundMetersPerSecond = 1540
pulseLengthMillimeters = pulseLength * speedOfSoundMetersPerSecond * 1e3
```

Performing this calculation we get an output of a pulse duration of 511 samples,
which comes out to a spatial pulse length of 3.15mm.

### Transducer Response

To make the transducer response as a gaussian-weighted sinusodial we can reuse
what we learned in the previous section:

```python
impulseResponse = signal.gausspulse(timeVec, fc=2.5*1e6, bw=0.4)
freqResponse, transducerFreqs = powerSpectrum(impulseResponse, fs)
```

Here's what the impulse- and frequency responses look like:

>> ![task1_transducer](figures/task1_transducer.png)
>> Transducer responses along with signals and spectra.
