import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components

st.set_option('deprecation.showPyplotGlobalUse', False)

from joblib import load

# Set page configuration
st.set_page_config(
   page_title="Insurance Cost Prediction",
   page_icon="💰",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Load the model and scaler
model = load('C:\\Users\\HP\\Desktop\\Med_Cost_Prediction\\Code\\Deployment\\model.pkl')
scaler = load('C:\\Users\\HP\\Desktop\\Med_Cost_Prediction\\Code\\Deployment\\scaler.pkl')
    
# Define the app
def run():
    st.title('Health Insurance Cost Prediction')

    # User inputs
    st.sidebar.header('User Input Features')
    age = st.sidebar.slider('Age', 18, 100, 30)
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 30.0)
    smoker = st.sidebar.selectbox('Smoker', ['yes', 'no'])
    children = st.sidebar.slider('Number of Children', 0, 5, 0)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    region = st.sidebar.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])
    
    # Convert user inputs into a dataframe
    input_data = pd.DataFrame({'age': [age], 'bmi': [bmi], 'smoker': [smoker], 'children': [children], 'sex': [sex], 'region': [region]})

    # Convert 'sex' to numeric
    input_data['sex'] = input_data['sex'].map({'male': 0, 'female': 1})
    
    # Convert 'smoker' to numeric
    input_data['smoker'] = input_data['smoker'].map({'no': 0, 'yes': 1})

    # Create dummy variables for 'region'
    region_dummies = pd.get_dummies(input_data['region'], prefix='region', drop_first=True)
    input_data = pd.concat([input_data.drop('region', axis=1), region_dummies], axis=1)
    
    # Manually add the missing dummy variables with a value of 0
    missing_cols = {'region_northwest', 'region_southeast', 'region_southwest'} - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0

    # List of feature names from the training data in the same order
    feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']

    # Reorder the columns of the input data
    input_data = input_data[feature_names]
    # Scale the input data using the same scaler used in training
    
    input_data_scaled = scaler.transform(input_data)

        # User Input Validation
    if bmi < 10 or bmi > 60:
        st.error('BMI should be between 10 and 60.')
    elif age < 18 or age > 100:
        st.error('Age should be between 18 and 100.')
    elif children < 0 or children > 5:
        st.error('Number of children should be between 0 and 5.')
    else:
        # Make predictions and show them on the app
        if st.button('Predict'):
            prediction = model.predict(input_data_scaled)
            st.success(f'Estimated Insurance Cost: {prediction[0]:.2f}')

            # Histogram of predicted insurance costs
            # fig, ax = plt.subplots()
            #sns.histplot(prediction, bins=50, ax=ax)
            #ax.set_title('Histogram of Predicted Insurance Costs')
            #st.pyplot(fig)

            # Bar chart of feature importances
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x=input_data.columns, y=model.feature_importances_, ax=ax)
                ax.set_title('Feature Importances')
                plt.xlabel('Features')
                plt.xticks(rotation=45)
                st.pyplot(fig)


    # Model Information
    st.sidebar.header('Model Information')
    st.sidebar.text(f'Model: {type(model).__name__}')
    st.sidebar.text(f'Model Parameters: {model.get_params()}')

    # Data Exploration
    st.header('🔍 Data Exploration')
    if st.checkbox('Show data histogram'):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(input_data, bins=50, ax=ax, color='skyblue', edgecolor='salmon')
        plt.title('Data Histogram', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        st.pyplot(fig)

    # Prediction Explanation
    # Note: You need to install the shap library (pip install shap)
    import shap

    # Prediction Explanation
    st.header('🔮 Prediction Explanation')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data_scaled)

    # Create the SHAP force plot and save it as an HTML file
    shap_force_plot = shap.force_plot(explainer.expected_value, shap_values, input_data)
    shap.save_html("shap_plot.html", shap_force_plot)

    # Display the HTML file in your Streamlit app
    with open("shap_plot.html", 'r', encoding='utf-8') as f:
        html_string = f.read()
        components.html(html_string, height=600)

if __name__ == '__main__':
    run()