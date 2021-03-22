import pandas as pd
from nltk.corpus import stopwords
import re

def preprocess(tweet):
    tweet = remove_mention(tweet)
    tweet = remove_url(tweet)
    tweet = alphanumerify(tweet)
    tweet = lowercase(tweet)
    tweet = remove_stopwords(tweet)

    print("preprocess")

def remove_mention(tweet):
    # Removing tweets with mentions of other users
    # The username will be removed with the @ and be replaced with an empty string
    return re.sub(r'@[a-zA-Z0-9_]+', '', tweet)

def remove_url(tweet):
    # Removing URLs
    removed_http = re.sub(r'https?://[a-zA-Z0-9_.]+', ' ', tweet)
    return re.sub(r'www\.[a-zA-Z0-9_.]+', ' ', removed_http)

def alphanumerify(tweet):
    # Removing any characters that are not alphabet or number
    # Only alphabets and number will remains, the rest will be replaced with a space
    return re.sub(r'[^a-zA-Z0-9]+', ' ', tweet)

def lowercase(tweet):
    # Not sure if this function is needed but it lowercase the string
    return tweet.lower()

def remove_stopwords(tweet):
    # Remove the stopwords generated from the nltk corpus library
    # List of stopwords
    STOPWORDS = stopwords.words('english')
    words = tweet.split()
    for word in words:
        if word in STOPWORDS:
            pattern = '(?:^|\W)' + word + '(?:$|\W)'
            tweet = re.sub(pattern, ' ', tweet)
    return tweet

def is_min_threshold(tweet, threshold):
    # Returns a boolean that reflect if the tweet meet the minimum threshold
    # Default threshold is 4
    threshold = 4
    words = tweet.split()
    return (len(words) > threshold)

################## FEATURES ##################
def caps_count(tweet):
    # Returns the number of words that are all capitalised in the tweet
    return re.subn(r'[A-Z]+\b', '', tweet)[1]

def exclamation_count(tweet):
    # Returns the number of times '!' appeared in the tweet
    return re.subn(r'!', '', tweet)[1]

def character_count(tweet):
    # Returns the number of words with unnecessary number of repeated characters (more than 2)
    return re.subn(r'(.)\1{2,}', '', tweet)[1]

def main():
    # NOTE: Need to split train and test set
    train = pd.read_csv('text_emotion.csv')
    # "tweet_id","sentiment","author","content"
    test_string = "@tiffanylue 123 word a he's what! ice-cream @tai_ping www.google.co https://aa http http://was.al.com wowwwww. I an am the is isnt it marvelous"
    print(alphanumerify(remove_mention(remove_url(test_string))))
    print(is_min_threshold(test_string, 4))
    print("main")
    a = 'ABC Aaba 123 DASD'
    print(caps_count(a))
    print(a)
    print(character_count('aaab sdas thiss is so coooool aba aaa'))

if __name__ == "__main__":
    main()