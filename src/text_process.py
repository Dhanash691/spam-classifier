import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords") #downloads the english stopwords list the first time you run the program

stemmer=PorterStemmer()
stop_words=set(stopwords.words("english"))
#we created this as reusable objects outside the function so programme can reuse them and does not have to recreate them every single time it process a message

def preprocess_text(text):
    text=text.lower() # converts text into alll lower case
    text=text.translate(str.maketrans("", "", string.punctuation)) #removes punctuation
    # make trans accepts three input (x,y,z) x-- charcters you want to replace, y--characters you want to replace them with, z--characters you want to delete entirely
    words=text.split()

    processed_words=[]
    
    for word in words:
        if word not in stop_words:
            processed_words.append(stemmer.stem(word)) #each word which is not a stop word(which we imported from ntlk library), is added to processed words after reducing the word to its root form using porter stemmer

    return " ".join(processed_words)

if __name__ == "__main__":
    sample = "Congratulations!! You have WON a FREE lottery."

    print(preprocess_text(sample))

