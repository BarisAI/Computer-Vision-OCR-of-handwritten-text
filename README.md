# IAM Word-Level Handwritten OCR

This repository contains a **word-level handwritten OCR system** built using deep learning and evaluated on the **IAM Handwriting Database**.

The goal of the project is to recognize handwritten **English words** from grayscale word images using a CNN–RNN architecture trained with **CTC loss**.

---

## Project Overview

- **Task:** Word-level handwritten text recognition (OCR)
- **Dataset:** IAM Handwriting Database (word-level)
- **Model:** CNN + BiLSTM + CTC
- **Framework:** PyTorch

The system takes a single handwritten word image as input and predicts the corresponding text sequence without requiring explicit character segmentation.

---

## Dataset

This project uses the **IAM Handwriting Database (word-level)**.

Due to the large size of the dataset, **raw word images are not included** in this repository.

Please download the dataset from the following Kaggle mirror:

`https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database`

The repository assumes the default directory structure provided by the Kaggle dataset.
Only the annotation file `archive/words_new.txt` and dataset split files are included here.

---

## Repository Structure
```
├── IAM_Word_Level_OCR.ipynb    # Main training and evaluation notebook
├── archive/
│   └── words_new.txt          # IAM word annotations
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── runs/                      # Training logs
├── best_ocr_model.pth
└── Report.pdf                 # Final project report
```

Input Word Image (200×200, grayscale)
```
→ ResNet18 (CNN encoder, ImageNet pretrained)
→ Feature map to sequence conversion
→ BiLSTM (2 layers)
→ Linear layer (character logits + blank)
→ CTC Loss (training) / Greedy decoding (inference)
```
CTC loss allows alignment-free training for variable-length word predictions.

---

## Training Details

- **Optimizer:** AdamW  
- **Learning rate:** 1e-4  
- **Batch size:** 32  
- **Scheduler:** ReduceLROnPlateau  
- **Loss:** CTC Loss  
- **Metrics:** CER, WER, Word Accuracy  

The best model is selected based on **lowest validation CER**.

---

## Results (Test Set)

- **CER:** 0.1166  
- **WER:** 0.3503  
- **Word Accuracy:** 0.6497  
- **Test samples:** 3825  

The model shows consistent performance between validation and test sets. Most errors occur for longer words or visually ambiguous characters.

---

## Requirements

Main libraries:
```
PyTorch
torchvision
NumPy
matplotlib
Pillow
TensorBoard
```
## How to Run

Download the dataset from kaggle.

Install dependencies

Open and run the notebook:
`IAM_Word_Level_OCR.ipynb`

## Notes

Trained model weights are not included due to file size limits.

The full training and evaluation pipeline can be reproduced using the provided notebook.

## Authors

**Baris Surmelioglu**

**Ozan Polat**
