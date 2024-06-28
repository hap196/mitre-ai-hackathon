import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean()}")

X_test = test_data.drop("id", axis=1)

test_data["phishy"] = model.predict(X_test)
