import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import StandardScaler
import joblib

test_data = pd.read_csv("test.csv")

tokenizer = BertTokenizer.from_pretrained("saved_model")
bert_model = TFBertModel.from_pretrained("saved_model")

def tokenize_urls(urls):
    return tokenizer(
        urls, padding=True, truncation=True, max_length=512, return_tensors="tf"
    )

def run_bert_on_data(urls):
    encoded_inputs = tokenize_urls(urls)
    outputs = bert_model(encoded_inputs)
    return outputs.last_hidden_state[:, 0, :]

bert_predictions = run_bert_on_data(test_data["url"].tolist())

bert_results = pd.DataFrame(
    bert_predictions.numpy(),
    columns=[f"bert_feature_{i}" for i in range(bert_predictions.shape[1])],
)
bert_results["id"] = test_data["id"]

def extract_url_features(url):
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_slashes"] = url.count("/")
    features["num_underscores"] = url.count("_")
    features["contains_gov"] = int(".gov" in url)
    features["contains_com"] = int(".com" in url)
    features["contains_org"] = int(".org" in url)
    return features


url_features = test_data["url"].apply(lambda url: pd.Series(extract_url_features(url)))
print("URL features extracted from entire dataset:\n", url_features.head())

print("Missing values in URL features:\n", url_features.isnull().sum())

url_features = url_features.fillna(-1)

test_data = pd.concat([test_data, url_features], axis=1)
print("Columns in test_data after concatenation:\n", test_data.columns)

feature_columns = (
    ["id"]
    + [f"bert_feature_{i}" for i in range(bert_predictions.shape[1])]
    + [
        "url_length",
        "num_dots",
        "num_hyphens",
        "num_slashes",
        "num_underscores",
        "contains_gov",
        "contains_com",
        "contains_org",
    ]
)

merged_data = pd.merge(
    test_data, bert_results, on="id", how="inner", suffixes=("", "_bert")
)

scaler = joblib.load("scaler.pkl")
print("Scaler loaded from 'scaler.pkl'.")

X_test = merged_data[feature_columns[1:]].fillna(-1)
X_test = scaler.transform(X_test)

input_shape = (X_test.shape[1],)
inputs = tf.keras.Input(shape=input_shape)

try:
    combined_model = tf.keras.models.load_model("trained_combined_model.h5")
    print("Combined model loaded successfully.")
except TypeError as e:
    print(f"Encountered an error while loading the model: {e}")
except Exception as e:
    print(f"Unexpected error occurred: {e}")

if "combined_model" in locals():
    predictions = combined_model.predict(X_test)

    final_predictions = pd.DataFrame(
        {
            "id": merged_data["id"],
            "phishy_pred": (predictions > 0.5).astype(int).flatten(),
        }
    )

    final_predictions.to_csv("final_predictions.csv", index=False)
    print("Final predictions saved to 'final_predictions.csv'.")
else:
    print("Model could not be loaded. Predictions were not made.")
