# Exercise 2 - Ultrasound Transmission and Reception

## Part 1

### Generating a Gaussian-weighted sinusoidal pulse

A Gaussian-weighted sinusoidal pulse - A "Gaussian pulse".

```python
def generateGaussianPulse(A, f0, fs, T, sigma):
    timeVector = np.arange(-T/2, T/2, 1/fs)
    gauss = np.exp(-timeVector**2 / (2 * sigma**2))
    sine = np.sin(2 * np.pi * f0 * timeVector)
    return A * gauss * sine
```

>> [task1_gauss](figures/task1_gauss.png")
>> Plot of gaussian pulse.
