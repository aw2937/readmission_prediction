# Patient Readmission Prediction using Neural Networks

This project demonstrates an end-to-end machine learning workflow for predicting hospital readmission risk (<30 days) for patients with diabetes, using a Neural Network built with TensorFlow/Keras.

**Goal:** Predict whether a patient is likely to be readmitted to the hospital within 30 days based on their demographic and clinical encounter data.

**Dataset:** [Diabetes 130-US hospitals for years 1999-2008 Data Set](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) from the UCI. Please download `diabetic_data.csv` and place it in the `data/raw/` directory.


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd patient-readmission-prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the dataset:** Obtain `diabetic_data.csv` from the UCI link above and place it in the `data/raw/` directory.

## How to Run

1.  **Data Preprocessing (Jupyter Notebook):**
    *   Open and run the `notebooks/1_Data_Exploration_and_Preprocessing.ipynb` notebook. This will perform EDA, clean the data, preprocess it (scaling, encoding), split it into train/validation/test sets, and save the processed files to `data/processed/`.

2.  **(Optional) Create SQLite Database:**
    *   If you want to demo the SQL loading capability, run the utility script:
        ```bash
        python src/db_utils.py
        ```
    *   You can then uncomment and run the relevant cell in Notebook 1 to load data from the DB.

3.  **Model Training (Python Script):**
    *   Run the training script from the project root directory:
        ```bash
        python src/train.py --config config/model_config.yaml
        ```
    *   This script loads the processed data, builds the NN model, trains it using parameters from the config file, applies class weights and callbacks (EarlyStopping, ModelCheckpoint), saves the best model to `results/saved_model/`, and saves the training history plot to `results/plots/`.

4.  **Model Evaluation (Python Script):**
    *   After training is complete, run the evaluation script:
        ```bash
        python src/evaluate.py --config config/model_config.yaml
        ```
    *   This script loads the *best saved model* and the *test set*, calculates performance metrics (AUC, classification report, confusion matrix), saves them to `results/metrics.json`, and saves a confusion matrix plot to `results/plots/`.

5.  **Interactive Exploration (Jupyter Notebook):**
    *   Open and run `notebooks/2_Model_Training_and_Evaluation.ipynb` to see the model building, training initiation, evaluation initiation, and results interpretation steps interactively. It leverages the functions from the `src` modules and loads the results generated by the Python scripts.

## Results Summary

The evaluation results (ROC AUC, Precision, Recall, F1-Score, Confusion Matrix) are saved in `results/metrics.json` and printed to the console by `evaluate.py`. Key plots (training history, confusion matrix) are saved in `results/plots/`.

The focus is on predicting the minority class ('Readmitted <30d'), so Recall for this class is a particularly important metric.
