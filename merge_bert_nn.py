import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import re
import math
from urllib.parse import urlparse, parse_qs

train_data = pd.read_csv('train.csv')

tokenizer = BertTokenizer.from_pretrained('saved_model')
bert_model = TFBertModel.from_pretrained('saved_model')

def tokenize_urls(urls):
    return tokenizer(urls, padding=True, truncation=True, max_length=512, return_tensors='tf')

def run_bert_on_data(urls):
    encoded_inputs = tokenize_urls(urls)
    outputs = bert_model(encoded_inputs)
    return outputs.last_hidden_state[:, 0, :]  

bert_predictions = run_bert_on_data(train_data['url'].tolist())

bert_results = pd.DataFrame(bert_predictions.numpy(), columns=[f'bert_feature_{i}' for i in range(bert_predictions.shape[1])])
bert_results['id'] = train_data['id']

bert_results.to_csv('bert_predictions.csv', index=False)
print("BERT predictions saved to 'bert_predictions.csv'.")

def extract_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['num_underscores'] = url.count('_')
    features['contains_gov'] = int('.gov' in url)
    features['contains_com'] = int('.com' in url)
    features['contains_org'] = int('.org' in url)
    features['tld'] = parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else ''
    features['num_query_params'] = len(query_params)
    features['contains_ip'] = int(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) is not None)
    features['contains_suspicious_word'] = int(any(word in url.lower() for word in ['login', 'verify', 'account', 'update', 'secure', 'signin', 'banking', 'password', 'confirm']))
    features['entropy'] = -sum((url.count(c) / len(url)) * math.log2(url.count(c) / len(url)) for c in set(url))
    features['path_length'] = len(parsed_url.path)
    features['query_length'] = len(parsed_url.query)
    features['num_special_chars'] = sum(1 for c in url if c in '@#%&')
    features['digit_to_letter_ratio'] = sum(c.isdigit() for c in url) / (sum(c.isalpha() for c in url) + 1)
    
    return features

url_features = train_data['url'].apply(lambda url: pd.Series(extract_url_features(url)))
url_features = pd.get_dummies(url_features, columns=['tld'])

print("URL features extracted from entire dataset:\n", url_features.head())

url_features = url_features.fillna(-1)

train_data = pd.concat([train_data, url_features], axis=1)
print("Columns in train_data after concatenation:\n", train_data.columns)

training_columns = list(url_features.columns)
joblib.dump(training_columns, "training_columns.pkl")
print("Training columns saved to 'training_columns.pkl'.")

feature_columns = ['id'] + [f'bert_feature_{i}' for i in range(bert_predictions.shape[1])] + training_columns

merged_data = pd.merge(train_data, bert_results, on='id', how='inner', suffixes=('', '_bert'))

merged_data.to_csv('merged_train_data_with_bert.csv', index=False)
print("Merged train data with BERT predictions saved to 'merged_train_data_with_bert.csv'.")

merged_train_data = pd.read_csv('merged_train_data_with_bert.csv')

feature_columns = [f'bert_feature_{i}' for i in range(768)] + training_columns

X = merged_train_data[feature_columns].fillna(-1)
y = merged_train_data['phishy']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to 'scaler.pkl'.")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {accuracy}')

model.save('trained_combined_model.h5')
print("Trained model saved to 'trained_combined_model.h5'.")
