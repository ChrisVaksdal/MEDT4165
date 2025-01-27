# MEDT4165 - Exercise 1

Written by: Christoffer-Robin Vaksdal

## Task 1 - Signal Generation and plotting

### Generating a signal

Our goal: to generate a continuous sinusodial signal over time using the following
parameters:

```python
f0 = 1MHz   # Signal Frequency
fs = 50MHz  # Sampling Frequency
T = 10us    # Total Duration
```

To generate the signal, we can use the python module Numpy to store our signal
as a `numpy.ndarray`. Using the parameters from earlier we can generate a
unit-amplitude signal as follows:

```python
signal = np.sin(2 * np.pi * f0 * np.arange(0, T, 1/fs))
```

The resulting signal looks like this:
> ![task1_signal](figures/task1_signal.png)
>> A Continuous sinusoidal signal with a frequency of 1MHz, sampled at 50MHz.

### Generating noise

Next, we want to generate random Gaussian noise with the same size and
sample-rate as our signal. Gaussian noise has a probability density function
(pdf) equal to the normal distribution, so we can use the built-in numpy
function `numpy.random.normal()` to generate unit-amplitude noise:

```python
noise = np.random.normal(0, 1, len(np.arange(0, T, 1/fs)))
```

The resulting noise signal looks like this:
> ![task1_noise](figures/task1_noise.png)
>> Random Gaussian noise with unit amplitude, sampled at 50MHz.

### Combining signal and noise

Using the signal- and noise-generation from earlier, we can create a combined
signal with noise present by simply adding the two together:

```python
noisySignal = signal * signalAmplitude + noise * noiseAmplitude
```

The resulting combined signal looks like this:
> ![task1_combined](figures/task1_combined.png)
>> A sinusoidal signal with unit-amplitude combined with random Gaussian noise
>> of the same unit-amplitude.

In the following figure, you can see three different sets of signal- and
noise-amplitude with individual plots showing the signal, the noise and the
combined noisy signal:
> ![task1_all_plots](figures/task1_all_plots.png)
>> Raw signals, raw noise and combined signals with different amplitudes.

In the above figure you can see three different scenarios. In the 1st, the
signal and noise have the same amplitude and the signal is quite hard to make
out. In the 2nd, the signal amplitude has been increased and as a result, the
signal is clearly visible in the combined plot. In the 3rd, the amplitude of the
noise is greater compared to the 2nd scenario, and as a result it is very hard
to make out the signal.

## Task 2 - Calculating the signal-to-noise ratio (SNR)

### SNR Calculation

Calculating the SNR is simple enough and can be achieved using the following
two python-functions (one for calculating the ratio, one for converting to dB):

```python
def calculateSnr(signal, noise):
    # SNR_est = mean(signal^2) / mean(noise^2)
    return np.mean(signal**2) / np.mean(noise**2)

def calculateSnrDb(signal, noise):
    # Decibel: SNR_est_dB = 10*log10(SNR_est) [dB]
    return 10 * np.log10(calculateSnr(signal, noise))
```

Using these functions and data from the previous task, we can calculate the
signal-to-noise ratios for the three different scenarios:

> ![task2_snr](figures/task2_snr.png)
>> Combined signals with calculated SNRs.

### Creating a target SNR parameter

Now, let's make a function that can scale the noise to achieve a desired target SNR:

```python
def getTargetScaledNoise(unitSignal, unitNoise, targetSnrDb):
    snrNoScaling = calculateSnr(unitSignal, unitNoise)

    # Scale ~ sqrt(SNR_est / 10^(SNR/10))
    scale = np.sqrt(snrNoScaling / 10**(TARGET_SNR_dB/10))
    return unitNoise * scale
```

With this function, we can create a new combined signal which should in theory
have the desired SNR

```python
TARGET_SNR_dB = 7
scaledNoise = getTargetScaledNoise(unitSignal, unitNoise, TARGET_SNR_dB)
snrScaledNoise = calculateSnrDb(unitSignal, scaledNoise)
```

The resulting SNR is very close to the target (7.00dB vs. 7.00dB).
To make sure the values don't get out of hand, we can normalize the signal by
dividing all the values in the combined signal by the peak-amplitude. The
resulting signal looks like this:

> ![task2_snr_normalized](figures/task2_snr_scaled_normalized.png)
>> Normalized plot of signal combined with noise with calculated SNRs.

## Task 3 - Power Spectrum (Frequency) Analysis

The power spectrum of a signal shows how the energy of the signal is
distributed between different frequency-components. It can be calculated using
the Fast Fourier Transform (FFT) which is conveniently available to us via
Numpy. Let's write a simple python function which takes in a signal and returns
the power spectrum (both axes. Amplitudes and frequencies):

```python
def calculatePowerSpectrum(signal):
    # P(f) = 1/N * abs(fft(signal, N)^2

    N = len(signal) # Number of samples

    spectrum = np.fft.fft(signal, N)

    powerSpectrum = (1/N) * np.abs(spectrum)**2
    powerSpectrum = np.fft.fftshift(powerSpectrum)  # Shift so 0Hz is in the center.

    psdb = 10 * np.log10(powerSpectrum)             # Convert to dB.

    freqs = np.fft.fftfreq(N, d=1/fs)
    freqs = np.fft.fftshift(freqs)                  # Shift so 0Hz is in the center.

    return psdb, freqs
```

Using this function, we can calculate the power spectrum of our signal. In
the below figure, you can see the power spectra of our raw sinusoidal signal
(unit amplitude signal from before) and that of the signal combined with our
scaled noise (normalized) from Task 2.

> ![task3_spectra](figures/task3_spectra.png)
>> Power spectra of raw signal and signal combined with noise, respectively

As you can see, both figures show a significant peak at &plusmn;1Hz. This is
expected, as this represents the sinusoidal signal itself. In the combined
signal, we see there is significantly more variance in the other frequency
components and we also see the height of the peak compared to the rest of the
signal (the noisy region at the bottom) is significantly lower compared to that
of the pure signal. In the pure sinusoidal signal, everything other than the
peaks representing our signal is down in the region of -600dB (basically
nothing). Meanwhile, in the combined signal, the noisy part resides around
-30dB. The distance between the peaks we care about and the noisy region at the
bottom is indicative of the SNR (in fact, it kind of *is* the SNR).

## Task 4 - Filtering to improve SNR

### Designing a bandpass filter

Using the SciPy-python module, we can construct a bandpass Butterworth filter
to attenuate the parts of our signal which we are not interested in. We can
also use SciPy (`signal.freqz()`) to calculate the frequency response of the
filter. By multiplying the numerator coefficient 'b', we can effectively
increase the gain of the filter.

```python
from scipy import signal

def getFilterCoeffs(fLow, fHigh, filterOrder, gain):
    b, a = signal.butter(filterOrder, [fLow, fHigh],
                    btype="bandpass", output="ba", fs=fs
    )
    b *= gain
    return b, a

fLow = f0 * 0.9
fHigh = f0 * 1.1
filterOrder = 5

b, a = getFilterCoeffs(fLow, fHigh, filterOrder, gain)

w, h = signal.freqz(b, a, fs=fs, worN=np.linspace(-fs/2, fs/2, 1000)) 
```

After creating the filter and applying the extra gain, the filter response
looks like this (here plotted against the signal power spectrum from earlier):

> ![task4_filter_response](figures/task4_filter_response.png)
>> Filter response of bandpass filter shown against signal power spectrum.

### Applying the filter to improve signal quality

First, let's generate a signal with very poor SNR using the target SNR
functionality from task2 and then try to apply a bandpass-filter to it and
see what we get. I decided to use `scipy.signal.filtfilt`, which applies the
filter twice (once in each direction) in order to preserve phase information.
Not really necessary in this case, but it's cool and produces a good result.

```python
time = generateTimeAxis()        
signal = generateUnitSignal()
noise = scaleNoiseToTargetSNR(signal, generateUnitNoise(), -10)

snrdBNoFilter = calculateSnrDb(signal, noise)   # Expected: -10

combined = normalizeSignal(signal + noise)

# I messed around with the parameters and these numbers work well
b, a = getFilterCoeffs(f0 * 0.8, f0 * 1.2, 6, 2)
filtered = signal.filtfilt(b, a, combined)
```

> ![task4_filtered_signal](figures/task4_filtered_signal.png)
>> Plot showing noisy signal along with filtered signal.

As you can see, the signal can be reconstructed! And with a great SNR to boast.
But what's going on? The signal looks dampened weirdly, almost as if some low
frequency component snuck it's way in or something. Let's try to apply a window
-function to see if that can help:

```python
from scipy.signal import windows
window = windows.hamming(N)
windowedSignal = combined * window
filteredWindowed = filtfilt(b, a, windowedSignal)
```

> ![task4_filtered_signal](figures/task4_filtered_windowed_signal.png)
>> Plot showing noisy signal along with windowed filtered signal.

That kind of worked a little bit, but there is still some dampening going on.
Next thing I would try would be to pad the signal (could use zeros or some
constant since we are already using a window-function, but it might be better
to use mirrored versions of the signal itself).
