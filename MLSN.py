import pandas as pd
from sklearn.model_selection import train_test_split

fake_df = pd.read_csv("Fake.csv/Fake.csv")
real_df = pd.read_csv("True.csv/True.csv")

fake_df["label"] = 0
real_df["label"] = 1

#relevant columns
use_cols = ["title", "text", "label"]
df = pd.concat([fake_df[use_cols], real_df[use_cols]], ignore_index=True)

# 80% train / 20% test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Fake % in Train:", train_df["label"].mean())
print("Fake % in Test:", test_df["label"].mean())
