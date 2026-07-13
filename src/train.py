from preprocess import load_and_clean
from text_process import preprocess_text
from sklearn.model_selection import train_test_split

df = load_and_clean("data/spam.csv")
df["transformed_text"]=df["message"].apply(preprocess_text) #preprocess_text function is applied on all the messages in message column and a new column transformed_text saves that

x=df["transformed_text"]
y=df["label"]

# print(df.head())
# print()
# print(df.columns)
# print()
# print(x.head())
# print()
# print(y.head())

x_train, x_test, y_train, y_test =train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("original dataset")
print(y.value_counts(normalize=True))

print("\nTraining Dataset")
print(y_train.value_counts(normalize=True))

print("\nTesting Dataset")
print(y_test.value_counts(normalize=True))