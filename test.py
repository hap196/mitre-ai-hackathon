import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

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

bert_classifier_model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

bert_classifier_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = bert_classifier_model.fit(
    {"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
)


def get_embeddings(model, input_ids, attention_mask):
    with tf.GradientTape() as tape:
        outputs = model.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings


bert_model = TFBertModel.from_pretrained("bert-base-uncased")
train_embeddings = get_embeddings(bert_model, X_train_ids, X_train_masks).numpy()
test_embeddings = get_embeddings(bert_model, X_test_ids, X_test_masks).numpy()

other_features_train = train_data[
    ["create_age_months", "expiry_age_months", "update_age_days"]
].values
other_features_test = test_data[
    ["create_age_months", "expiry_age_months", "update_age_days"]
].values

X_train_combined = np.hstack((train_embeddings, other_features_train))
X_test_combined = np.hstack((test_embeddings, other_features_test))

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_combined, labels_train)

test_predictions = clf.predict(X_test_combined)

test_data["phishy"] = test_predictions

submission = test_data[["id", "phishy"]]
submission.to_csv("submission.csv", index=False)

bert_classifier_model.save_pretrained("model")
tokenizer.save_pretrained("tokenizer")
