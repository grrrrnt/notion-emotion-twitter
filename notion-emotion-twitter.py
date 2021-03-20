import pandas as pd
from nltk.corpus import stopwords
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

def remove_url(tweet):
    # Removing URLs
    removed_http = re.sub(r'https?://[a-zA-Z0-9_.]+', ' ', tweet)
    return re.sub(r'www\.[a-zA-Z0-9_.]+', ' ', removed_http)

def lowercase(tweet):
    # Not sure if this function is needed but it lowercase the string
    return tweet.lower()

def remove_stopwords(tweet):
    ## List of stopwords
    STOPWORDS = stopwords.words('english')
    words = tweet.split()
    for word in words:
        if word in STOPWORDS:
            pattern = '(?:^|\W)' + word + '(?:$|\W)'
            tweet = re.sub(pattern, ' ', tweet)
    return tweet

def main():
    # NOTE: Need to split train and test set
    train = pd.read_csv('text_emotion.csv')
    # "tweet_id","sentiment","author","content"
    test_string = "@tiffanylue 123 word he's what! ice-cream @tai_ping www.google.co https://aa http http://was.al.com wowwwww. I an am the is isnt it marvelous"
    print(remove_stopwords(alphanumerify(remove_mention(remove_url(test_string)))))

    print("main")

if __name__ == "__main__":
    main()