import pandas as pd
import nltk
import re

from langdetect import detect

## Method to check if a given text is in English
## NOTE: Very short-text can give unexpected results, emoticons should be removed before calling this method
def isEnglish(text):
    language = detect(text)
    if (language == en):
        return True
    else:
        return False

def preprocess(tweets):
    print("preprocess")
    
def main():
    # NOTE: Need to split train and test set
    train = pd.read_csv('text_emotion.csv')
    print("main")

if __name__ == "__main__":
    main()