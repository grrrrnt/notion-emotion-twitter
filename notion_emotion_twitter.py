import os
import re

import pandas as pd
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import fasttext

class CS4248BestClass:
    ABBREV_CSV = pd.read_csv('abbreviations.csv')
    ABBREV_DICT = dict(zip(ABBREV_CSV.abbreviation, ABBREV_CSV.replacement))
    WORD_PATTERN = re.compile('\w+')
    STOPWORDS = stopwords.words('english')
    PRETRAINED_MODEL_PATH = 'lid.176.ftz'
    LANGUAGE_MODEL = fasttext.load_model(PRETRAINED_MODEL_PATH)
    SPELL = SpellChecker(distance=1)

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
            if not self.is_english(tweet):
                continue
            tweet = self.correct_spelling(tweet)
            data.append((tweet, features))
        return data

    def is_english(self, tweet):
        ## Removal of all non-english tweets
        ## NOTE: Remove all short-text, shorten repeated words and remove emoticons for higher accuracies
        language_prediction = self.LANGUAGE_MODEL.predict(tweet)[0][0]
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
        return self.WORD_PATTERN.sub(self.replace_word, tweet)

    def remove_stopwords(self, tweet):
        # Remove the stopwords generated from the nltk corpus library
        return self.WORD_PATTERN.sub(self.remove_stopword, tweet)

    def is_min_threshold(self, tweet, threshold=4):
        # Returns a boolean that reflect if the tweet meet the minimum threshold
        words = tweet.split()
        return len(words) >= threshold

    def correct_spelling(self, tweet):
        # Replaces misspelled words with corrected spellings
        # # Using spellchecker (seems to perform better than textblob & autocorrect libraries)
        words = tweet.split()
        tweet = " ".join([self.SPELL.correction(word) for word in words])
        return tweet

    def replace_word(self, matchobj):
        # Helper function to replace abbreviations
        word = matchobj.group(0)
        if word in self.ABBREV_DICT:
            return self.ABBREV_DICT[word]
        else:
            return word

    def remove_stopword(self, matchobj):
        # Helper function to remove stopwords
        word = matchobj.group(0)
        if word in self.STOPWORDS:
            return ''
        else:
            return word

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

    CHAR_REPLACEMENTS = [(re.compile(pattern), replacement)
            for pattern, replacement in [(r'([Hh][Aa]){2,}', r'haha'), (r'[Ll]([Oo][Ll]){2,}', r'lol')]]
    CHAR_PATTERN = re.compile(r'([A-Za-z])\1{2,}')
    def character_count(self, tweet):
        # Returns the number of words with unnecessary number of repeated characters (more than 2)
        # Shortens repeated characters into single character
        for pattern, new in self.CHAR_REPLACEMENTS:
            tweet = pattern.sub(new, tweet)
        return self.CHAR_PATTERN.subn(r'\1', tweet)

    def train_embeddings(self, content):
        content_file_name = 'text_content.txt'
        tweets = self.preprocess(content)
        with open(content_file_name, 'w') as f:
            f.writelines(tweet for tweet, _ in tweets)
        # According to fastText API https://fasttext.cc/docs/en/python-module.html
        # Default parameters are skipgram, dimensions = 100, window size = 5
        model = fasttext.train_unsupervised(content_file_name)
        os.remove(content_file_name)
        return model

    def main(self):
        # NOTE: Need to split train and test set
        train = pd.read_csv('text_emotion_sample.csv')
        content_train = train['content']
        sentiment_train = train['sentiment']

        tweets = self.preprocess(content_train)
        for tweet in tweets:
            print(tweet)    # (tweet, features)

        model = self.train_embeddings(pd.read_csv('text_emotion.csv').content)
        print(model.get_nearest_neighbors('friday'))

if __name__ == "__main__":
    CS4248BestClass().main()
