# Customer Churn Prediction with Keras MLP and Streamlit

## Overview

This repository implements a customer churn prediction model using a Keras MLP.
The model predicts customer churn based on various profile features. 
Streamlit is used for deploying the model in an interactive and user-friendly manner.

## Project Structure

- **.idea**: Configuration files for the IDE. Ignore for version control.
- **Assignment3.ipynb**: Jupyter Notebook with initial model development.
- **label_encoders.pkl**: Serialized label encoders for categorical features.
- **main.py**: Streamlit application for model deployment.
- **mlp_model.h5**: Trained Keras model saved in HDF5 format.
- **mpl_model_kt.h5**: Additional Keras model file (verify necessity and remove if not needed).
- **requirements.txt**: Python dependencies for running the application.
- **scaler.pkl**: Serialized scaler for numerical features.
- **templates**: Folder containing HTML templates for the Streamlit app.

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/PapaYaw-Boampong/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application:**

    ```bash
    streamlit run main.py
    ```

    Visit the provided local URL (e.g., http://localhost:8501) in your browser to access the Streamlit app and make predictions.

## Model Training

The customer churn prediction model is built using a Keras MLP. Training involves labeled data with features such as gender,	Partner,
Dependents,	PhoneService,	MultipleLines,	etc.


## Link to Deployed model Demo
```https://youtu.be/ZofUujaghmE
    ```
