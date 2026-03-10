# Voice Activity Detection and Voiced/Unvoiced Classification

This project was developed for the **COE 216 Signals and Systems course**.

## Project Overview

The goal of this project is to detect speech activity within an audio recording and classify speech segments into voiced and unvoiced sounds.

The system consists of two main stages:

### 1. Voice Activity Detection (VAD)

Speech segments are detected using:

- Short-Time Energy
- Dynamic Noise Threshold
- 20 ms frame window
- 50% overlap
- Hamming window

Silent regions are removed to create a shortened audio output.

### 2. Voiced vs Unvoiced Detection

Speech frames are classified using:

- Short-Time Energy
- Zero Crossing Rate (ZCR)

Classification rules:

Voiced → High Energy + Low ZCR  
Unvoiced → Low/Medium Energy + High ZCR

## Files

- `vad.py` → Voice Activity Detection implementation
- `voiced_unvoiced.py` → Voiced and Unvoiced classification
- `ses_kaydi.wav` → Input audio file
- `odev_cikti_grafigi.png` → Generated analysis graph

## Technologies Used

- Python
- NumPy
- SciPy
- Librosa
- Matplotlib

## Course

COE 216 – Signals and Systems  
Spring 2025–2026
