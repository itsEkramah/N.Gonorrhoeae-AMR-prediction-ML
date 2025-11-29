# Neisseria gonorrhoeae AMR Prediction

This project implements a machine learning pipeline to predict antibiotic resistance in *Neisseria gonorrhoeae* using genomic unitig data and metadata.

## Project Structure

- `DATA/`: Contains the dataset (metadata and unitig Rtab files).
- `src/`: Source code for data processing, model training, and evaluation.
    - `data_processing.py`: Functions to load and preprocess data.
    - `models.py`: Model definitions and training logic.
    - `evaluation.py`: Visualization and evaluation metrics.
- `main_analysis.ipynb`: Jupyter Notebook for running the full analysis.
- `requirements.txt`: Python dependencies.

## Setup

1. **Install Dependencies**
   Ensure you have Python 3.8+ installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data**
   Ensure the `DATA` directory contains:
   - `metadata.csv`
   - `azm_sr_gwas_filtered_unitigs.Rtab`
   - `cip_sr_gwas_filtered_unitigs.Rtab`
   - `cfx_sr_gwas_filtered_unitigs.Rtab`

## Running the Analysis

Open the Jupyter Notebook `main_analysis.ipynb` and run all cells:
```bash
jupyter notebook main_analysis.ipynb
```

You can select the antibiotic to analyze by changing the `ANTIBIOTIC` variable in the "Data Loading" section (options: 'Azithromycin', 'Ciprofloxacin', 'Cefixime').

## Models Implemented

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting (XGBoost/sklearn)

## Evaluation

The pipeline evaluates models using 5-fold Stratified Cross-Validation and reports:
- Accuracy
- Precision
- Recall
- F1-Score

It also generates Confusion Matrices, ROC Curves, and Feature Importance plots.
