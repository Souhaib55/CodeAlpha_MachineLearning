# ============================================================
#  CodeAlpha ML Internship — Run All 4 Tasks
#  Run this script from the project folder in PowerShell:
#      .\run_all_tasks.ps1
# ============================================================

$python = ".\.venv312\Scripts\python.exe"
$ErrorActionPreference = "Continue"

function Banner($title) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Yellow
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

# ─── TASK 1: Credit Scoring ────────────────────────────────
Banner "TASK 1 — Credit Scoring Model"
Write-Host @"
  Dataset : UCI German Credit (auto-downloaded if needed)
  Fallback: synthetic data if download fails
  Models  : Logistic Regression, Decision Tree, Random Forest
"@ -ForegroundColor Green

# The german_credit_data.csv in Datasets/credit has no target column.
# Task 1 will auto-download the proper german.data from UCI instead.
& $python task1_credit_scoring.py --model all
if ($LASTEXITCODE -ne 0) {
    Write-Host "[TASK 1] Script exited with code $LASTEXITCODE" -ForegroundColor Red
}

# ─── TASK 2: Emotion Recognition ───────────────────────────
Banner "TASK 2 — Emotion Recognition From Speech (EMO-DB)"
Write-Host @"
  Dataset : EMO-DB wav files in Datasets\wav\
  Features: MFCC (40 coefficients, max_len=200)
  Model   : CNN (TensorFlow/Keras)
"@ -ForegroundColor Green

& $python task2_emotion_recognition_speech.py `
    --data-dir "Datasets\wav" `
    --dataset emodb `
    --epochs 30 `
    --batch-size 16 `
    --max-files 0
if ($LASTEXITCODE -ne 0) {
    Write-Host "[TASK 2] Script exited with code $LASTEXITCODE" -ForegroundColor Red
}

# ─── TASK 3: Handwritten Character Recognition ─────────────
Banner "TASK 3 — Handwritten Character Recognition (MNIST)"
Write-Host @"
  Dataset : MNIST (auto-downloaded by TensorFlow)
  Model   : CNN (TensorFlow/Keras)
  Epochs  : 10
"@ -ForegroundColor Green

& $python task3_handwritten_character_recognition.py `
    --dataset mnist `
    --epochs 5 `
    --report
if ($LASTEXITCODE -ne 0) {
    Write-Host "[TASK 3] Script exited with code $LASTEXITCODE" -ForegroundColor Red
}

# ─── TASK 4: Disease Prediction ────────────────────────────
Banner "TASK 4 — Disease Prediction (Breast Cancer built-in)"
Write-Host @"
  Dataset : Breast Cancer (sklearn built-in, no download needed)
  Models  : Logistic Regression, Random Forest, SVM, XGBoost
"@ -ForegroundColor Green

& $python task4_disease_prediction.py --model all
if ($LASTEXITCODE -ne 0) {
    Write-Host "[TASK 4] Script exited with code $LASTEXITCODE" -ForegroundColor Red
}

Banner "ALL TASKS COMPLETE"
Write-Host "Check the output above for classification reports and metrics." -ForegroundColor Green
