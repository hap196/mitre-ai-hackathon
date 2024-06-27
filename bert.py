import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Handle missing values
train_data.fillna(-1, inplace=True)
test_data.fillna(-1, inplace=True)

# Extract URLs and labels
urls_train = train_data["url"].tolist()
labels_train = train_data["phishy"].tolist()
urls_test = test_data["url"].tolist()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Tokenize the URLs
def tokenize_urls(urls, tokenizer, max_length=128):
    return tokenizer(
        urls, max_length=max_length, padding=True, truncation=True, return_tensors="tf"
    )


# Tokenize the train and test data
train_encodings = tokenize_urls(urls_train, tokenizer)
test_encodings = tokenize_urls(urls_test, tokenizer)

# Extract the input_ids and attention_masks
X_train_ids = train_encodings["input_ids"]
X_train_masks = train_encodings["attention_mask"]
X_test_ids = test_encodings["input_ids"]
X_test_masks = test_encodings["attention_mask"]

# Convert labels to TensorFlow tensors
y_train = tf.convert_to_tensor(labels_train)

# Load the BERT model
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    {"input_ids": X_train_ids, "attention_mask": X_train_masks},
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.2,
)

# Predict on the test set
test_predictions = model.predict(
    {"input_ids": X_test_ids, "attention_mask": X_test_masks}
)
test_data["phishy"] = tf.argmax(test_predictions.logits, axis=1).numpy()

# Create the submission file
submission = test_data[["id", "phishy"]]
submission.to_csv("submission.csv", index=False)
