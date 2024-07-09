import numpy as np 
import pandas as pd 
import os
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC

# Reading and splitting the data
train_df = pd.read_csv("../merged_train_data_with_bert.csv")
eval_df = pd.read_csv("../test.csv")
train_df_y = train_df.pop('phishy')

print("good")

# BERT
from transformers import BertTokenizer, TFBertForSequenceClassification

train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

train_data.fillna(-1, inplace=True)
test_data.fillna(-1, inplace=True)

urls_train = train_data["url"].tolist()
labels_train = train_data["phishy"].tolist()
urls_test = test_data["url"].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_urls(urls, tokenizer, max_length=128):
    return tokenizer(
        urls, max_length=max_length, padding=True, truncation=True, return_tensors="tf"
    )

train_encodings = tokenize_urls(urls_train, tokenizer)
test_encodings = tokenize_urls(urls_test, tokenizer)

X_train_ids = train_encodings["input_ids"]
X_train_masks = train_encodings["attention_mask"]
X_test_ids = test_encodings["input_ids"]
X_test_masks = test_encodings["attention_mask"]

y_train = tf.convert_to_tensor(labels_train)

# Code to have checkpoints on the model training
checkpoint_path = "../model.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Use the Adam optimizer directly from tf.keras.optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(
    {"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[cp_callback]
)

test_predictions = model.predict(
    {"input_ids": X_test_ids, "attention_mask": X_test_masks}
)
test_data["phishy"] = tf.argmax(test_predictions.logits, axis=1).numpy()

# Preprocessing: adding new variables based on the URL
def count_char(s, char):
    return s.count(char)

def count_num(s):
    return sum(c.isdigit() for c in s)

# Adding the length of the URL
train_df['url_len'] = train_df['url'].apply(len)
train_df['url_nums'] = train_df['url'].apply(lambda x: count_num(x))
train_df['num_slashes'] = train_df['url'].apply(lambda x: count_char(x, '/'))
train_df['num_periods'] = train_df['url'].apply(lambda x: count_char(x, '.'))
train_df['num_hyphens'] = train_df['url'].apply(lambda x: count_char(x, '-'))
train_df['contains_com'] = train_df['url'].str.contains('.com')
train_df['contains_gov'] = train_df['url'].str.contains('.gov')
train_df['contains_org'] = train_df['url'].str.contains('.org')
train_df['contains_html'] = train_df['url'].str.contains('.html')
train_df['contains_www'] = train_df['url'].str.contains('www.')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(train_df, train_df_y, test_size=0.3, random_state=42)

# Preprocessing 2: removing identifier columns and scaling numerical values
X_train = X_train.drop(['id', 'url'], axis=1)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
boolean_cols = X_train.select_dtypes(include=['bool']).columns

# Define the preprocessing steps for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a ColumnTransformer to apply the transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('bool', 'passthrough', boolean_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = Sequential()
model.add(Dense(32, activation='relu'))  # Input layer with 3 input features
model.add(Dense(16, activation='relu'))  # Hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=BinaryCrossentropy(),
              metrics=['accuracy', AUC()])

history = model.fit(X_train_processed, y_train, epochs=30, batch_size=7, validation_split=0.3, verbose=1)

results = model.evaluate(X_test_processed, y_test)
print(f'Test Loss: {results[0]}')
print(f'Test Accuracy: {results[1]}')
print(f'Test AUC: {results[2]}')

# Preprocessing: adding new variables based on the URL
test_df = pd.read_csv('../merged_test_data_with_bert.csv')

results_df = test_df.pop('id')
# Removing identifier columns and URL column
test_df = test_df.drop(['url'], axis=1)

# Identify numerical and categorical columns
numerical_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = test_df.select_dtypes(include=['object']).columns
boolean_cols = test_df.select_dtypes(include=['bool']).columns

# Define the preprocessing steps for numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a ColumnTransformer to apply the transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('bool', 'passthrough', boolean_cols)
    ])

test_df_processed = preprocessor.fit_transform(test_df)

probabilities = model.predict(test_df_processed)
predictions = (probabilities >= 0.5).astype(int)
predictions = np.concatenate(predictions)

final_results = pd.DataFrame({
    'id': results_df,
    'pred': predictions
})

print(final_results)
final_results.to_csv('final_preds.csv', index=False)
