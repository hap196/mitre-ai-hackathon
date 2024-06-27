import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.fillna(-1, inplace=True)
test_data.fillna(-1, inplace=True)

def extract_features(df):
    df["url_length"] = df["url"].apply(len)
    df["num_dots"] = df["url"].apply(lambda x: x.count("."))
    df["num_slashes"] = df["url"].apply(lambda x: x.count("/"))
    df["num_hyphens"] = df["url"].apply(lambda x: x.count("-"))
    return df

train_data = extract_features(train_data)
test_data = extract_features(test_data)

train_data.drop("url", axis=1, inplace=True)
test_data.drop("url", axis=1, inplace=True)

X = train_data.drop(["id", "phishy"], axis=1)
y = train_data["phishy"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data.drop("id", axis=1))

model = Sequential(
    [
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(
    X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val)
)

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy}")

test_data["phishy"] = model.predict(X_test).round().astype(int)

submission = test_data[["id", "phishy"]]
submission.to_csv("submission.csv", index=False)
