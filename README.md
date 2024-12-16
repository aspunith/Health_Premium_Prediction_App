# Health_Premium_Prediction_App

## Project Overview
The **Health Premium Prediction App** is a machine learning-powered tool designed to predict health insurance premiums based on customer attributes such as age, income level, marital status, number of dependents, insurance plan type, and genetic risk factors. This project offers valuable insights to insurance companies and individuals by identifying patterns in premium pricing.

---

## Motivation
The motivation for this project stems from a curiosity to analyze and uncover the patterns in which people buy insurance policies. This includes understanding the influence of factors such as:
- Age
- Income Level
- Insurance Plan Type
- Marital Status
- Number of Dependents
- Genetical Risk Factors

This project also served as an opportunity to practice advanced machine learning techniques, error analysis, and model segmentation for improving predictions.

---

## Features
1. **Prediction of Health Premiums**: Based on user input, the app provides an accurate estimate of the premium.
2. **Interactive Web Interface**: A user-friendly Streamlit-based application.
3. **Dynamic Segmentation**: Separate models for different age groups to enhance accuracy.

---

## Approach and Methodology
1. **Initial Dataset**:
   - The project started with a single dataset containing premiums and user demographics.
   - Features such as age, income level, marital status, and number of dependents were analyzed.

2. **Error Analysis**:
   - Initial predictions showed significant errors for individuals aged ≤ 25.
   - These errors highlighted the need for additional features and a more tailored model approach.

3. **Model Segmentation**:
   - The dataset was divided into two groups: **age ≤ 25** and **age > 25**.
   - An additional feature, **genetical_risk**, was introduced for the younger age group.
   - Separate models were trained for each segment, resulting in improved accuracy.

4. **Modeling and Deployment**:
   - Machine learning algorithms were implemented using Scikit-learn and XGBoost.
   - The final models were deployed using Streamlit for an interactive user experience.

---

## Dataset
The dataset contains anonymized information about individuals, including:
- Age
- Income Level
- Marital Status
- Number of Dependents
- Genetical Risk Factors
- Insurance Plan Type

**In-Depth Analysis**:
A detailed analysis of the dataset is available in the [Jupyter Notebook](https://github.com/aspunith/Health_Premium_Prediction_App) under the `Notebooks and Datasets` section.

---

## Tech Stack
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- **Model Deployment**: Streamlit
- **Version Control**: Git and GitHub

---

## How to Use
### Prerequisites
1. Install Python 3.8 or higher.
2. Clone the repository:
   ```bash
   git clone https://github.com/aspunith/Health_Premium_Prediction_App.git
   ```
3. Navigate to the project directory:
   ```bash
   cd Health_Premium_Prediction_App
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Launch the Streamlit app:
   ```bash
   streamlit run main.py
   ```
2. Open the application in your browser at `http://localhost:8501`.

3. Enter user details to predict health insurance premiums.

---

## Repository Structure
```
Health_Premium_Prediction_App/
├── artifacts/
│   ├── model_young.joblib         # Model for age ≤ 25
│   ├── model_rest.joblib          # Model for age > 25
│   ├── scaler_young.joblib        # Scaler for age ≤ 25
│   └── scaler_rest.joblib         # Scaler for age > 25
├── datasets/
│   └── health_premium_data.csv    # Dataset used for training
├── notebooks/
│   └── analysis.ipynb             # Detailed EDA and insights
├── main.py                        # Streamlit application
├── prediction_helper.py           # Helper functions for prediction
└── requirements.txt               # Dependencies
```

---

## Results and Insights
- **Pattern Identification**:
  - Younger individuals (age ≤ 25) showed highly varying premiums due to genetic risks and other factors.
  - Segmented models improved prediction accuracy significantly.

- **Model Accuracy**:
  - The segmented models outperformed the single model approach, reducing errors by 15% for younger age groups.

---

## Future Scope
1. Incorporate more features such as health history and geographic location.
2. Experiment with deep learning techniques for better predictions.
3. Extend the app to provide insights into policy recommendations based on user data.

## Live Demo

Explore the application live on Streamlit: [Health Premium Prediction App](https://healthpremiumnprediction.streamlit.app/)

---

## Author
**A S Punith**  
Software Engineer | Machine Learning Enthusiast  
[LinkedIn](https://www.linkedin.com/in/aspunith) | [GitHub](https://github.com/aspunith)

---


