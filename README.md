# Transaction Anomaly Detector

This project detects anomalous financial transactions using a neural network classifier built with TensorFlow and Keras.

## Structure

- `data/` - place your dataset CSV here (e.g., transactions.csv)
- `models/` - model training and saving scripts
- `utils/` - preprocessing and evaluation functions
- `app/` - Streamlit dashboard for demo and visualization
- `config/` - database connection config (MongoDB)

## Setup

1. Create and activate your virtual environment.
2. Install requirements: `pip install -r requirements.txt`
3. Place your dataset CSV file inside `data/` folder.
4. Run model training: `python models/train_model.py`
5. Launch app: `streamlit run app/dashboard.py`

## Notes

- MongoDB is used for storing transaction data.
- TensorFlow/Keras is used for the MLP anomaly detector.
- Streamlit serves a user-friendly interactive interface.

## Refernces 

<img src="https://github.com/CharanTeja-BS/transaction-anomaly-detector/blob/main/Screenshot%202025-05-22%20221630.png" alt="Project Image 1" width="600"/>

<img src="https://github.com/CharanTeja-BS/transaction-anomaly-detector/blob/main/Screenshot%202025-05-22%20221655.png" alt="Project Image 2" width="600"/>


# Thank YOU 🫶
