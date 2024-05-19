
# Vehicle Insurance Fraud Detection Using Machine Learning

This project includes a machine learning model developed to detect and prevent vehicle insurance fraud, along with a Streamlit interface to visualize the model's results. This repository contains the source code for the project, the data used for training the model, and the necessary files to create the interface.

## Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## About the Project

Vehicle insurance fraud constitutes a significant cost for insurance companies and can also harm customers. This project employs machine learning techniques to detect fraudulent insurance claims. The model analyzes various features to detect fraud with high accuracy, and presents this analysis through a user-friendly Streamlit interface.

## Features

- **Machine Learning Model:** A trained model for fraud detection.
- **Data Processing:** Cleaning, processing, and preparing data for the model.
- **Streamlit Interface:** A web-based interface to visualize model results and interact with users.
- **Comprehensive Analysis:** Displays fraud probabilities and model performance metrics.

## Installation

To run the project on your local machine, follow these steps.

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/username/vehicle-insurance-fraud.git
    cd vehicle-insurance-fraud
    ```

2. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

## Usage

Once the Streamlit interface is launched, you can access it through the local server that opens in your browser. Through the interface, you can upload insurance claim data and view the model's predictions on these claims.

## Project Structure

```
vehicle-insurance-fraud/
├── data/
│   └── dataset.csv            # Training and testing data
├── models/
│   └── fraud_detection_model.pkl  # Trained model file
├── app.py                     # Streamlit application file
├── requirements.txt           # Required Python libraries
├── README.md                  # Project description file
└── src/

```


