# Predictive Modeling of Insurance Medical Costs Using Regression Techniques

This project is a comprehensive analysis and prediction of medical insurance costs based on a variety of factors. The goal is to create a model that can accurately predict insurance costs for a given individual based on their personal information.

## Project Overview

The project uses a dataset that includes the following features:

- Age: The age of the individual.
- Sex: The gender of the individual.
- BMI: The Body Mass Index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
- Children: The number of children the individual has.
- Smoker: Whether or not the individual is a smoker.
- Region: The individual's residential area in the US, northeast, southeast, southwest, northwest.

The target variable is:

- Charges: The individual's medical costs billed by health insurance.

## Methodology

The project involves several steps:

1. **Data Preprocessing**: The data is cleaned and preprocessed. This involves handling missing values, outliers, and categorical variables.

2. **Exploratory Data Analysis (EDA)**: EDA is performed to understand the distribution of data, the relationship between different variables, and find any interesting observations.

3. **Model Building**: Several regression models are built and their performance is compared. The models include linear regression, decision tree regression, random forest regression, and more.

4. **Model Evaluation**: The models are evaluated using appropriate metrics and the best performing model is selected.

5. **Model Interpretation**: The predictions of the model are interpreted using SHAP values. This helps in understanding the contribution of each feature towards the prediction.

## Deployment

The model is deployed as a web application using Streamlit. The user can input their details and the application will predict their medical insurance costs.

## Future Work

Future work includes improving the model performance by trying out different models, feature engineering, and hyperparameter tuning. Also, the web application can be improved by adding more features and making it more user-friendly.