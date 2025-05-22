import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.preprocessing import preprocess_data
from config import db_config
from pymongo import MongoClient

# Load model and scaler
MODEL_PATH = "models/mlp_model.h5"
SCALER_PATH = "models/scaler.save"

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_mongo_collection():
    client = MongoClient(db_config.MONGO_URI)
    db = client[db_config.DATABASE_NAME]
    collection = db[db_config.COLLECTION_NAME]
    return collection

def fetch_data_from_db(collection):
    data = list(collection.find())
    if len(data) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Drop MongoDB ID for modeling
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    return df

def main():
    st.title("Transaction Anomaly Detector Dashboard")

    model, scaler = load_resources()

    collection = get_mongo_collection()

    df = fetch_data_from_db(collection)

    if df.empty:
        st.warning("No transaction data found in MongoDB collection.")
        return

    st.write("### Sample Transactions")
    st.dataframe(df.head())

    # Preprocess for prediction
    X = df.drop(columns=['is_anomaly'])
    X_scaled = scaler.transform(X)

    st.write("### Predict Anomalies")
    pred_prob = model.predict(X_scaled)
    pred_labels = (pred_prob > 0.5).astype(int)

    df['Predicted Anomaly'] = pred_labels

    st.dataframe(df[['is_anomaly', 'Predicted Anomaly']].head(20))

    st.write("### Anomaly Distribution")
    fig = px.histogram(df, x='Predicted Anomaly', color='Predicted Anomaly', barmode='group')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
