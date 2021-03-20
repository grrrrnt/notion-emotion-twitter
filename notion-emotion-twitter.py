import pandas as pd
import nltk
import re

def preprocess(tweets):
    print("preprocess")

def remove_mention(tweet):
    # Removing tweets with mentions of other users
    # The username will be removed with the @ and be replaced with an empty string
    return re.sub(r'@[a-zA-Z0-9_]+', '', tweet)

def alphanumerify(tweet):
    # Removing any characters that are not alphabet or number
    # Only alphabets and number will remains, the rest will be replaced with a space
    return re.sub(r'[^a-zA-Z0-9]+', ' ', tweet)

def main():
    # NOTE: Need to split train and test set
    train = pd.read_csv('text_emotion.csv')
    # "tweet_id","sentiment","author","content"
    test_string = "@tiffanylue 123 word he's what! ice-cream @tai_ping"
    print(alphanumerify(remove_mention(test_string)))

    print("main")

if __name__ == "__main__":
    main()