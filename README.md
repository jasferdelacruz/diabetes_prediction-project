# Diabetes Prediction System 🩺💻

## Overview

The Diabetes Prediction System is an intelligent web application that leverages machine learning to assess an individual's risk of developing diabetes. Built using Python and Streamlit, this application provides a user-friendly interface for quick and insightful health risk assessment.


## 🌟 Key Features

- **Interactive User Interface**: Easy-to-use web application with intuitive sliders
- **Machine Learning Prediction**: Uses Random Forest Classifier for accurate predictions
- **Comprehensive Visualizations**: Comparative scatter plots of health parameters
- **Personalized Results**: Custom input and personalized risk assessment
- **High Accuracy**: Model trained on the Pima Indians Diabetes Dataset

## 🔬 How It Works

The application analyzes eight key health indicators:
1. Number of Pregnancies
2. Glucose Level
3. Blood Pressure
4. Skin Thickness
5. Insulin Level
6. Body Mass Index (BMI)
7. Diabetes Pedigree Function
8. Age

Users input their personal health data through an interactive sidebar, and the machine learning model provides a risk assessment within seconds.

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip (Python Package Manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-system.git
   cd diabetes-prediction-system
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Running the Application

```bash
streamlit run app.py
```

## 📊 Dataset

The application uses the **Pima Indians Diabetes Dataset**, which contains medical predictor variables and a target variable of diabetes outcome.

### Dataset Characteristics
- Total Instances: 768
- Features: 8 medical predictors
- Target: Diabetes outcome (0 = No Diabetes, 1 = Diabetes)

## 🤖 Machine Learning Model

### Random Forest Classifier
- Ensemble learning method
- Handles non-linear relationships
- Reduces overfitting
- Provides feature importance insights

## 📈 Model Performance

- **Accuracy**: Dynamically calculated during runtime
- Trained using 80% of the dataset
- Tested on 20% of the dataset

## 🚨 Disclaimer

**Important**: This is a predictive tool for educational purposes and should **NOT** replace professional medical advice. Always consult healthcare professionals for accurate medical diagnosis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Name- Jasfer Dela Cruz and Francis Jelo Nieves

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- youtube videos
