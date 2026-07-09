# 🩺 Diabetes Prediction using Machine Learning & Deep Learning

This repository contains a comprehensive healthcare project to predict diabetes based on patient diagnostic measurements. The model has been trained on a dataset of 100,000 records.

## 🚀 Live Demo
The project has been deployed as an interactive web application using Streamlit:
👉 **[Click here to test the Live App](https://diabetes-prediction-by-me.streamlit.app/)**

---

## 📊 Project Overview
- **Machine Learning:** Implemented a tuned **Random Forest Classifier** using a pipeline with SMOTE and RandomUnderSampler to handle extreme class imbalance.
- **Deep Learning:** Built a custom **Artificial Neural Network (ANN)** using PyTorch with Batch Normalization and Dropout layers to prevent overfitting.
- **Web Interface:** Built a clean UI using **Streamlit** that takes patient data (Age, BMI, HbA1c level, etc.) and gives real-time predictions using both models.

## 📈 Model Performance
Here is how the models performed on the validation set:

| Model | Accuracy | Recall (Class 1) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Random Forest Pipeline** | 93.79% | 83.93% | 0.70 |
| **PyTorch ANN** | 93.23% | 86.54% | 0.69 |

> 💡 **Key Insight:** Since this is a medical use case, **Recall** is our primary metric to minimize False Negatives (missing a diabetic patient). The PyTorch ANN achieved the highest recall of **86.54%**.

---

## 📁 Repository Structure
- `app.py`: Streamlit application code for the web interface.
- `diabetes-predicition-with-ml-and-dl.ipynb`: The complete Kaggle notebook containing EDA, preprocessing, and training code.
- `random_forest_model.pkl`: The serialized best-performing Random Forest pipeline.
- `diabetes_ann_model.pt`: Saved weights of the PyTorch ANN model.
- `requirements.txt`: List of dependencies required to run the project.

## 🛠️ Local Installation & Setup
To run this app on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/imanshrajsingh-boost/Diabetes-Prediction.git](https://github.com/imanshrajsingh-boost/Diabetes-Prediction.git)
   cd Diabetes-Prediction
