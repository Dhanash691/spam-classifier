import joblib

from src.text_process import preprocess_text

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")



def predict_message(message):
    #process the input message
    processed=preprocess_text(message)

    #convert text to TF_IDF vector
    vector=vectorizer.transform([processed])

    #Predict class
    prediction=model.predict(vector)

    #predict probabilities
    probabilities = model.predict_proba(vector)
    confidence = probabilities.max() * 100

    #convert nuerical prediction to text
    label = "SPAM" if prediction[0] == 1 else "HAM"

    return label, confidence

def main():
    message = input("Enter a message: ")

    label, confidence = predict_message(message)

    print("\nPrediction :", label)
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()


