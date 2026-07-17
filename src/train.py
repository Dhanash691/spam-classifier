from src.preprocess import load_and_clean
from src.text_process import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

df = load_and_clean("data/spam.csv")
df["transformed_text"]=df["message"].apply(preprocess_text) #preprocess_text function is applied on all the messages in message column and a new column transformed_text saves that

X=df["transformed_text"]
Y=df["label"]

# print(df.head())
# print()
# print(df.columns)
# print()
# print(x.head())
# print()
# print(y.head())

X_train, X_test, Y_train, Y_test =train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# print("original dataset")
# print(y.value_counts(normalize=True))

# print("\nTraining Dataset")
# print(y_train.value_counts(normalize=True))

# print("\nTesting Dataset")
# print(y_test.value_counts(normalize=True))

#TF_IDF VECTORIZATION

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train) #the model is trained from only the training the data and then transformed into numbers?vectors
X_test = vectorizer.transform(X_test)

# print(type(X_train))
# print(X_train.shape)
# print(X_test.shape)

model=MultinomialNB()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
# print(Y_pred[:10])
# print(Y_test[:10].values)

#save the model
joblib.dump(model, "models/spam_model.pkl")

#save the vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


# print("Model saved successfully")
# print("Vectorizer saved successfully")

#EVALUATION

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# print("Accuracy :", accuracy_score(Y_test, Y_pred))
# print("Precision:", precision_score(Y_test, Y_pred))
# print("Recall   :", recall_score(Y_test, Y_pred))
# print("F1 Score :", f1_score(Y_test, Y_pred))

# print("\nConfusion Matrix")
# print(confusion_matrix(Y_test, Y_pred))