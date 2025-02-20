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
