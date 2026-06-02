# Speech Emotion Recognition - Phase 3

Final Phase 3 implementation for BIL216 Signals and Systems Final Project.

## Final Result

- Final Model: Optimized SVM with RBF Kernel
- Accuracy: 86.76%
- Best Parameters: C = 10, gamma = 0.001
- Dataset Size: 679 audio recordings
- Classes: Angry, Happy, Neutral, Sad, Surprised

## Feature Set

MFCC, Delta MFCC, Delta-Delta MFCC, ZCR, RMS/STE, Pitch statistics, Pitch Range, Spectral Centroid, Spectral Rolloff, Spectral Bandwidth, Spectral Flatness, Chroma, Spectral Contrast, Tonnetz, Duration, Silence Ratio, Kurtosis, and Skewness.

## Live Demo

The `demo.py` script demonstrates the final Phase 3 model on a selected WAV file.  
It trains the optimized SVM model using the extracted Phase 3 feature set and predicts the emotional class of a given audio file.

Example:

```bash
python demo.py
