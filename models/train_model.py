import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.preprocessing import load_data, preprocess_data
from utils.evaluation import evaluate_model, plot_confusion

DATA_PATH = os.path.join("data", "transactions.csv")
MODEL_PATH = os.path.join("models", "mlp_model.h5")
SCALER_PATH = os.path.join("models", "scaler.save")

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("Building model...")
    model = build_model(X_train.shape[1])

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    print("Evaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    evaluate_model(y_test, y_pred)
    plot_confusion(y_test, y_pred)

    print("Saving model and scaler...")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
