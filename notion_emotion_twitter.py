import re

import pandas as pd
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import fasttext

PRETRAINED_MODEL_PATH = 'lid.176.ftz'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

class CS4248BestClass:
    def __init__(self):
        csv = pd.read_csv('abbreviations.csv')
        self.patterns = [(re.compile(r'\b' + abbrev + r'\b', re.IGNORECASE), replace)
                for abbrev, replace in zip(csv.abbreviation, csv.replacement)]

    ################## PREPROCESSING ##################

    def preprocess(self, tweets):
        data = []
        for tweet in tweets:
            if not self.is_min_threshold(tweet, 4):
                continue
            tweet = self.remove_mention(tweet)
            tweet = self.remove_url(tweet)
            tweet, features = self.extract_features(tweet)
            tweet = self.alphanumerify(tweet)
            tweet = self.lowercase(tweet)
            tweet = self.replace_abbrev(tweet)
            tweet = self.remove_stopwords(tweet)
            if not self.is_english:
                print("not english")
                continue
            tweet = self.correct_spelling(tweet)
            data.append((tweet, features))
        return data

    def is_english(self, tweet):
        ## Removal of all non-english tweets
        ## NOTE: Remove all short-text, shorten repeated words and remove emoticons for higher accuracies
        language_prediction = model.predict(tweet)[0][0][0]
        return language_prediction == "__label__en"

    def remove_mention(self, tweet):
        # Removing tweets with mentions of other users
        # The username will be removed with the @ and be replaced with an empty string
        return re.sub(r'@[a-zA-Z0-9_]+', '', tweet)

    def remove_url(self, tweet):
        # Removing URLs
        removed_http = re.sub(r'https?://[a-zA-Z0-9_.]+', ' ', tweet)
        return re.sub(r'www\.[a-zA-Z0-9_.]+', ' ', removed_http)

    def alphanumerify(self, tweet):
        # Removing any characters that are not alphabet or number
        # Only alphabets and number will remains, the rest will be replaced with a space
        return re.sub(r'[^a-zA-Z0-9]+', ' ', tweet)

    def lowercase(self, tweet):
        # Not sure if this function is needed but it lowercase the string
        return tweet.lower()

    def replace_abbrev(self, tweet):
        # Replaces abbreviations/slang with full words/phrases
        for pattern, replacement in self.patterns:
            tweet = pattern.sub(replacement, tweet)
        return tweet

    def remove_stopwords(self, tweet):
        # Remove the stopwords generated from the nltk corpus library
        # List of stopwords
        STOPWORDS = stopwords.words('english')
        words = tweet.split()
        for word in words:
            if word in STOPWORDS:
                pattern = '(?:^|\W)' + word + '(?:$|\W)'
                tweet = re.sub(pattern, ' ', tweet)
        return tweet

    def is_min_threshold(self, tweet, threshold=4):
        # Returns a boolean that reflect if the tweet meet the minimum threshold
        words = tweet.split()
        return len(words) >= threshold

    def correct_spelling(self, tweet):
        # Replaces misspelled words with corrected spellings
        # # Using spellchecker (seems to perform better than textblob & autocorrect libraries)
        spell = SpellChecker()
        words = tweet.split()
        tweet = " ".join([spell.correction(word) for word in words])
        return tweet

    ################## FEATURES ##################

    def extract_features(self, tweet):
        features = {}
        features['caps'] = self.caps_count(tweet)
        features['exclamation'] = self.exclamation_count(tweet)
        tweet, features['character'] = self.character_count(tweet)
        return tweet, features

    def caps_count(self, tweet):
        # Returns the number of words that are all capitalised in the tweet
        return re.subn(r'[A-Z]+\b', '', tweet)[1]

    def exclamation_count(self, tweet):
        # Returns the number of times '!' appeared in the tweet
        return re.subn(r'!', '', tweet)[1]

    def character_count(self, tweet):
        # Returns the number of words with unnecessary number of repeated characters (more than 2)
        # Shortens repeated characters into single character
        return  re.subn(r'([A-Za-z])\1{2,}', r'\1', tweet)

    def main(self):
        # NOTE: Need to split train and test set
        train = pd.read_csv('text_emotion_sample.csv')
        content_train = train['content']
        sentiment_train = train['sentiment']

        tweets = self.preprocess(content_train)
        for tweet in tweets:
            print(tweet)    # (tweet, features)

if __name__ == "__main__":
    CS4248BestClass().main()
