import pandas as pd

#load the data (This defines 'df' so puthon knows what it is!)
df=pd.read_csv("spam.csv", encoding="latin-1")

#renaming column 1 and column 2 because they do not have proper names(I AM MODIFYING THE SAME DF HERE, INSTEAD OF SAVING TO NEW DATAFRAME)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)


#Deleting the empty columns(AGAIN MODIFYING)
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)

#print the first few rows to verify it worked
print(df.head())

#count number of columns
print(df["label"].value_counts())

#ML preprocessing(CONVERTING SPAM AND HAM TO BINARY)
df['label']=df['label'].map({'ham':0, 'spam':1})

#print
print(df.head())