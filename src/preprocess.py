import pandas as pd

def load_and_clean(path):
#load the data (This defines 'df' so puthon knows what it is!)
    df=pd.read_csv(path, encoding="latin-1")
 
#renaming column 1 and column 2 because they do not have proper names(I AM MODIFYING THE SAME DF HERE, INSTEAD OF SAVING TO NEW DATAFRAME)
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

#Deleting the empty columns(AGAIN MODIFYING)
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], errors="ignore", inplace=True)

    df=df[["label", "message"]]
#count number of columns
    # print(df["label"].value_counts())

#ML preprocessing(CONVERTING SPAM AND HAM TO BINARY)
    df['label']=df['label'].map({'ham':0, 'spam':1})

#remove Remove rows with missing labels/messages
    df = df.dropna(subset=["label", "message"])

# Remove duplicate messages
    df.drop_duplicates(inplace=True)


    return df

if __name__ == "__main__":
    df = load_and_clean("data/spam.csv")
    print(df["label"].value_counts())
    print()
    print(df.info())
    print()
    print(df["label"].value_counts())