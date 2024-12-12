# Diabetes Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon=":medical_thermometer:",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.big-title {
    font-size: 50px !important;
    font-weight: bold;
    color: #2C3E50;
    text-align: center;
    margin-bottom: 30px;
}
.subtitle {
    font-size: 20px !important;
    color: #34495E;
    text-align: center;
    margin-bottom: 20px;
}
.stApp {
    background-color: #F0F4F8;
}
.sidebar .sidebar-content {
    background-color: #E6F2FF;
}
</style>
""", unsafe_allow_html=True)

# Main Title and Introduction
st.markdown('<div class="big-title">Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predictive Machine Learning Application for Early Diabetes Detection</div>', unsafe_allow_html=True)

# Informative Introduction
st.markdown("""
### About the System
This web application uses machine learning to predict the likelihood of diabetes based on various health parameters. 
By analyzing key medical indicators, we provide a preliminary assessment to help individuals understand their potential risk.

### How It Works
1. Enter your personal and medical information in the sidebar
2. Our Random Forest Classifier will analyze your data
3. Receive a personalized diabetes risk prediction
""")

# Load the dataset
df = pd.read_csv(r'C:\Users\Jasfer\Desktop\diabetes_prediction-project\diabetes_prediction-project\Pima-Indians-diabetes-Dataset.csv')

# Sidebar for User Input
st.sidebar.header('Patient Information')

# Add Patient Name Input
patient_name = st.sidebar.text_input('Enter Patient Name', '')

# Existing slider inputs with descriptive text
def user_report():
    st.sidebar.subheader('Medical Parameters')
    pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 17, 0, help='Total number of pregnancies')
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 100, help='Plasma glucose concentration')
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70, help='Diastolic blood pressure (mm Hg)')
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20, help='Triceps skin fold thickness (mm)')
    insulin = st.sidebar.slider('Insulin Level', 0, 846, 79, help='2-Hour serum insulin (mu U/ml)')
    bmi = st.sidebar.slider('Body Mass Index (BMI)', 0.0, 67.0, 25.0, help='Body mass index (weight in kg/(height in m)^2)')
    diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.5, help='Diabetes heredity likelihood')
    age = st.sidebar.slider('Age', 21, 81, 35, help='Age of the patient')

    # Prepare user report data
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Prepare data for machine learning
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Collect user data
user_data = user_report()

# Train the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Prediction
user_result = rf.predict(user_data)
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100

# Display Results
st.header('Prediction Results')

# Personalize result with name
if patient_name:
    st.subheader(f'Results for {patient_name}')

# Diabetes prediction output
if user_result[0] == 0:
    st.success('🟢 Low Risk: Based on the analysis, you are not likely to have diabetes.')
else:
    st.error('🔴 High Risk: The analysis suggests a potential risk of diabetes. Please consult a healthcare professional.')

# Display accuracy
st.info(f'Model Accuracy: {accuracy:.2f}%')

# Visualization Section
st.header('Comparative Health Visualizations')

# Color selection based on prediction
color = 'red' if user_result[0] == 1 else 'blue'

# Create visualization functions to reduce code repetition
def create_scatter_plot(x_col, y_col, palette, title):
    fig = plt.figure(figsize=(10, 6))
    ax1 = sns.scatterplot(x=x_col, y=y_col, data=df, hue='Outcome', palette=palette)
    ax2 = sns.scatterplot(x=user_data[x_col], y=user_data[y_col], s=200, color=color)
    plt.title(title)
    st.pyplot(fig)

# Create multiple visualizations
visualizations = [
    ('Age', 'Pregnancies', 'Greens', 'Pregnancy Count vs Age'),
    ('Age', 'Glucose', 'magma', 'Glucose Levels vs Age'),
    ('Age', 'BloodPressure', 'Reds', 'Blood Pressure vs Age'),
    ('Age', 'SkinThickness', 'Blues', 'Skin Thickness vs Age'),
    ('Age', 'Insulin', 'rocket', 'Insulin Levels vs Age'),
    ('Age', 'BMI', 'rainbow', 'Body Mass Index vs Age'),
    ('Age', 'DiabetesPedigreeFunction', 'YlOrBr', 'Diabetes Pedigree Function vs Age')
]

for x_col, y_col, palette, title in visualizations:
    create_scatter_plot(x_col, y_col, palette, title)

# Footer
st.markdown('---')
st.markdown('**Disclaimer:** This is a predictive tool and should not replace professional medical advice.')