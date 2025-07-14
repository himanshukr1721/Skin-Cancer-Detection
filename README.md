# Skin Cancer Prediction Project

A machine learning project to predict skin cancer using patient data.
Deployed Link :- https://skinsafe-ai.streamlit.app/
## About

This project uses machine learning to predict skin cancer based on patient information and clinical features. It compares different algorithms and saves the best performing model.

## What it does

- Analyzes patient data to predict skin cancer
- Tests 7 different machine learning models
- Picks the best model based on performance
- Saves the trained model for future use

## Models Used

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes

## Files Created

After running the code, you'll get:
- `skin_cancer_model.pkl` - The trained model
- `scaler.pkl` - Data preprocessing scaler
- `label_encoders.pkl` - Categorical data encoders

## How to Use

1. **Install required libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

2. **Run the training code:**
   ```python
   python your_training_script.py
   ```

3. **Use the saved model:**
   ```python
   import joblib
   
   # Load the model
   model = joblib.load('skin_cancer_model.pkl')
   scaler = joblib.load('scaler.pkl')
   
   # Make predictions on new data
   prediction = model.predict(new_data)
   ```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Important Note

This is for educational purposes only. Do not use for actual medical diagnosis. Always consult a doctor for medical advice.

## Results

The code will show:
- Performance of each model
- Confusion matrix
- ROC curve
- Best model selection

The best model is automatically saved for future use.
