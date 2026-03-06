import numpy as np
import matplotlib.pyplot as plt

# DTMF frequency table
dtmf_freqs = {
    '1': (697, 1209),
    '2': (697, 1336),
    '3': (697, 1477),
    '4': (770, 1209),
    '5': (770, 1336),
    '6': (770, 1477),
    '7': (852, 1209),
    '8': (852, 1336),
    '9': (852, 1477),
    '*': (941, 1209),
    '0': (941, 1336),
    '#': (941, 1477)
}

# Sampling parameters
fs = 8000
duration = 0.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Select a key
key = '5'

# Get the two frequencies for the selected key
f1, f2 = dtmf_freqs[key]

# Generate DTMF signal
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title(f"DTMF Signal for Key '{key}'")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
