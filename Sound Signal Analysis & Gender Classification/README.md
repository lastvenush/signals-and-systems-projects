Audio Emotion Classification Project

This project implements an audio-based emotion classification system developed for the Signals and Systems course. The model extracts features from audio signals (e.g., MFCC) and uses a machine learning algorithm to classify emotions.

📊 Dataset

The dataset used in this project was created collaboratively:
Each group contributed approximately 18 audio samples
Total dataset size: 700+ audio files
The dataset is not included in this repository due to size and privacy considerations

Dataset Structure

To run the project, create a data/ folder and organize your audio files as follows:

data/

 ├── happy/
 
 ├── sad/
 
 ├── neutral/

 Requirements
 
Install the required libraries before running the project:
pip install -r requirements.txt

How to Run

Run the main script:
python main.py

Model Details

Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)
Classification Algorithm: (fill this based on your model — e.g., Random Forest, SVM, etc.)


📌 Notes

Dataset must be provided by the user
Make sure audio files are in supported formats (e.g., .wav)
Ensure correct folder structure before running the code
