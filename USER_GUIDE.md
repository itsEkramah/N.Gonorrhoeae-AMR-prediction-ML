# User Guide: Neisseria gonorrhoeae AMR Prediction

This guide will help you run the project step-by-step. It is designed for users with a biological background who may be new to coding.

## 1. Project Overview

This project uses machine learning to predict whether a bacteria sample (*Neisseria gonorrhoeae*) is resistant to antibiotics based on its DNA (unitigs).

We use five different models to make these predictions:
1.  **Logistic Regression**: A simple statistical model.
2.  **Random Forest**: A collection of decision trees.
3.  **SVM (Support Vector Machine)**: Finds the best boundary between resistant and susceptible samples.
4.  **XGBoost**: An advanced gradient boosting method.
5.  **CatBoost**: Another advanced boosting method that handles categorical data well.

## 2. Files to Keep vs. Delete

### **KEEP These Files** (Essential for the project)
*   **`DATA/`**: This folder contains your raw data (`metadata.csv` and `.Rtab` files). **Do not delete.**
*   **`src/`**: This folder contains the Python code that does the heavy lifting.
    *   `data_processing.py`: Loads and cleans the data.
    *   `models.py`: Defines the machine learning models (including XGBoost and CatBoost).
    *   `evaluation.py`: Creates the graphs and calculates accuracy.
*   **`main_analysis.ipynb`**: The interactive notebook where you run the analysis and see results.
*   **`requirements.txt`**: A list of software libraries needed to run the code.
*   **`README.md`**: General project documentation.

### **DELETE These Files** (Temporary or no longer needed)
You can safely delete these files as they were for testing during development:
*   `verify_pipeline.py` (We used this to check if the code works, but `main_analysis.ipynb` is what you need now).
*   `debug_output.txt`
*   `debug_output_2.txt`
*   `debug_indices.py`

## 3. How to Run the Project

### Step 1: Install Software (If not already installed)
You need **Python** and **Jupyter Notebook**. The easiest way is to install **Anaconda**.
1.  Download Anaconda for your OS (Linux/Windows/Mac).
2.  Install it following the instructions.

### Step 2: Set Up the Environment
Open your terminal (or Anaconda Prompt) and navigate to the project folder:
```bash
cd /path/to/N.Gonorrhoeae-AMR-prediction-ML
```

Install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Analysis
1.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  A web page will open. Click on **`main_analysis.ipynb`**.
3.  In the notebook, you will see cells with code.
4.  Go to the menu at the top and click **Cell > Run All**.
5.  Scroll down to see the results!

### Step 4: Changing the Antibiotic
To analyze a different antibiotic (e.g., Azithromycin instead of Ciprofloxacin):
1.  Find the cell in `main_analysis.ipynb` that says:
    ```python
    ANTIBIOTIC = 'Ciprofloxacin'
    ```
2.  Change it to:
    ```python
    ANTIBIOTIC = 'Azithromycin'
    ```
    (or `'Cefixime'`)
3.  Re-run the notebook (**Cell > Run All**).

## 4. Understanding the Results

*   **Accuracy**: Percentage of correct predictions.
*   **F1-Score**: A balanced measure of accuracy (useful if one class is rare).
*   **Confusion Matrix**: Shows how many resistant samples were correctly predicted vs. missed.
*   **Feature Importance**: The bar chart at the end shows which DNA sequences (unitigs) are most strongly linked to resistance. You can blast these sequences to find the genes.

## 5. Troubleshooting
*   **"ModuleNotFoundError"**: Means a library is missing. Run `pip install -r requirements.txt` again.
*   **"FileNotFoundError"**: Check that your `DATA` folder has the correct files.
