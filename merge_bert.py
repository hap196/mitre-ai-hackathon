import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_slashes'] = url.count('/')
    features['num_underscores'] = url.count('_')
    features['contains_gov'] = int('.gov' in url)
    features['contains_com'] = int('.com' in url)
    features['contains_org'] = int('.org' in url)
    return features

url_features = train_data['url'].apply(lambda url: pd.Series(extract_url_features(url)))
print("URL features extracted from entire dataset:\n", url_features.head())

print("Missing values in URL features:\n", url_features.isnull().sum())

url_features = url_features.fillna(-1)

train_data = pd.concat([train_data, url_features], axis=1)
print("Columns in train_data after concatenation:\n", train_data.columns)


feature_columns = ['id'] + [f'bert_feature_{i}' for i in range(bert_predictions.shape[1])] + \
                  ['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'num_underscores', 'contains_gov', 'contains_com', 'contains_org']

merged_data = pd.merge(train_data, bert_results, on='id', how='inner', suffixes=('', '_bert'))

merged_data.to_csv('merged_train_data_with_bert.csv', index=False)
print("Merged train data with BERT predictions saved to 'merged_train_data_with_bert.csv'.")

merged_train_data = pd.read_csv('merged_train_data_with_bert.csv')