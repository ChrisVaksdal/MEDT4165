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
> ![task1_signal](task1_signal.png)
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
> ![task1_noise](task1_noise.png)
>> Random Gaussian noise with unit amplitude, sampled at 50MHz.

### Combining signal and noise

Using the signal- and noise-generation from earlier, we can create a combined
signal with noise present by simply adding the two together:

```python
noisySignal = signal * signalAmplitude + noise * noiseAmplitude
```

The resulting combined signal looks like this:
> ![task1_combined](task1_combined.png)
>> A sinusoidal signal with unit-amplitude combined with random Gaussian noise
>> of the same unit-amplitude.

In the following figure, you can see three different sets of signal- and
noise-amplitude with individual plots showing the signal, the noise and the
combined noisy signal:
> ![task1_all_plots](task1_all_plots.png)
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

> ![task2_snr](task2_snr.png)
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
The new signal looks like this:

> ![task2_snr](task2_snr_scaled.png)
>> Combined signals with calculated SNRs.

## Task 3 - Power Spectrum (Frequency) Analysis

ASDASD
