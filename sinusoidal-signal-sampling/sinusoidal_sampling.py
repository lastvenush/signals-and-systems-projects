import numpy as np
import matplotlib.pyplot as plt

# Frequencies of signals
f1 = 5
f2 = 10
f3 = 15

# Continuous time axis
t = np.linspace(0, 1, 1000)

# Generate sinusoidal signals
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)
x3 = np.sin(2 * np.pi * f3 * t)

# Sum of signals
x_sum = x1 + x2 + x3

# Sampling frequency (Nyquist)
fs = 2 * max(f1, f2, f3)
ts = np.arange(0, 1, 1/fs)

# Sampled signals
xs1 = np.sin(2 * np.pi * f1 * ts)
xs2 = np.sin(2 * np.pi * f2 * ts)
xs3 = np.sin(2 * np.pi * f3 * ts)
xs_sum = xs1 + xs2 + xs3

# Plot continuous signals
plt.figure(figsize=(10,8))

plt.subplot(4,1,1)
plt.plot(t, x1)
plt.title("Continuous Signal 1")

plt.subplot(4,1,2)
plt.plot(t, x2)
plt.title("Continuous Signal 2")

plt.subplot(4,1,3)
plt.plot(t, x3)
plt.title("Continuous Signal 3")

plt.subplot(4,1,4)
plt.plot(t, x_sum)
plt.title("Sum of Continuous Signals")

plt.tight_layout()
plt.show()

# Plot sampled signals
plt.figure(figsize=(10,8))

plt.subplot(4,1,1)
plt.stem(ts, xs1)
plt.title("Sampled Signal 1")

plt.subplot(4,1,2)
plt.stem(ts, xs2)
plt.title("Sampled Signal 2")

plt.subplot(4,1,3)
plt.stem(ts, xs3)
plt.title("Sampled Signal 3")

plt.subplot(4,1,4)
plt.stem(ts, xs_sum)
plt.title("Sum of Sampled Signals")

plt.tight_layout()
plt.show()
