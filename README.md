
# CodeAlpha ML Internship Tasks

This repository contains four machine learning tasks implemented as separate Python scripts, plus a guide explaining the approach and code structure.

## Tasks and scripts

- Task 1: Credit Scoring Model -> task1_credit_scoring.py
- Task 2: Emotion Recognition From Speech -> task2_emotion_recognition_speech.py
- Task 3: Handwritten Character Recognition -> task3_handwritten_character_recognition.py
- Task 4: Disease Prediction From Medical Data -> task4_disease_prediction.py

See the detailed guide in Internship Guide .md.

## Setup

Use Python 3.9+.

Recommended installs:

pip install numpy pandas scikit-learn tensorflow librosa soundfile

Optional:

pip install tensorflow-datasets xgboost

## Running the tasks

Task 1 (Credit Scoring):

python task1_credit_scoring.py
python task1_credit_scoring.py --data path/to/credit.csv --model rf

Task 2 (Speech Emotion Recognition):

python task2_emotion_recognition_speech.py --data-dir path/to/RAVDESS --dataset ravdess
python task2_emotion_recognition_speech.py --data-dir path/to/TESS --dataset tess

Task 3 (Handwritten Character Recognition):

python task3_handwritten_character_recognition.py --dataset mnist
python task3_handwritten_character_recognition.py --dataset emnist --train-limit 20000 --test-limit 5000

Task 4 (Disease Prediction):

python task4_disease_prediction.py
python task4_disease_prediction.py --data path/to/medical.csv --target outcome --model svm

## Notes

- Datasets are not included. Download them separately and provide paths where needed.
- For Task 2, the script caches MFCC features to speed up repeated runs.
- For Task 1, if the German Credit dataset cannot be downloaded, a synthetic fallback is used.
