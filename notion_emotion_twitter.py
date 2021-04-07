import os
import re
import pprint
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import fasttext
from nrclex import NRCLex
from nltk.util import ngrams
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing

## Models import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class CS4248BestClass:
    ABBREV_CSV = pd.read_csv('abbreviations.csv')
    ABBREV_DICT = dict(zip(ABBREV_CSV.abbreviation, ABBREV_CSV.replacement))
    WORD_PATTERN = re.compile('\w+')
    STOPWORDS = stopwords.words('english')
    PRETRAINED_MODEL_PATH = 'lid.176.ftz'
    LANGUAGE_MODEL = fasttext.load_model(PRETRAINED_MODEL_PATH)
    SPELL = SpellChecker(distance=1)
    CV = CountVectorizer()
    TFID = TfidfTransformer(sublinear_tf=True)

    ################## PREPROCESSING ##################

    def preprocess(self, X, y):
        data = list(zip(X, y))
        preprocessed = []
        for tweet, sentiment in data:
            if not self.is_min_threshold(tweet, 4):
                # print(tweet + " -- Tweet removed due to word threshold")
                continue
            tweet = self.remove_mention(tweet)
            tweet = self.remove_url(tweet)
            tweet, features = self.extract_features(tweet)
            tweet = self.alphanumerify(tweet)
            tweet = self.lowercase(tweet)
            tweet = self.replace_abbrev(tweet)
            tweet = self.remove_stopwords(tweet)
            tweet = self.correct_spelling(tweet)
            if not self.is_english(tweet):
                # print(tweet + " -- Tweet removed due non-English")
                continue
            preprocessed.append((tweet, sentiment, features))
        return preprocessed

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
        features['lexicon'] = self.emotion_lexicon_score(tweet)
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

    def emotion_lexicon_score(self, tweet):
        # Returns a dictionary representing sum of word emotion
        words = tweet.split()
        tweet_emotion = {
            'anticipation': 0,
            'joy': 0,
            'negative': 0,
            'sadness': 0,
            'disgust': 0,
            'positive': 0,
            'anger': 0,
            'surprise': 0,
            'fear': 0,
            'trust': 0
        }
        lemmatizer = WordNetLemmatizer()
        for word in words:
            emotion = NRCLex(lemmatizer.lemmatize(word))
            word_emotion = emotion.raw_emotion_scores
            for key, value in word_emotion.items():
                tweet_emotion[key] += value
        return tweet_emotion

    def generate_ngram(self, tweet, n):
        # Return a list of all the ngrams in a tweet
        tokens = [token for token in tweet.split(" ") if token != ""]
        return list(ngrams(tokens, n))

    def TFIDF(self, tweet, sentiment):
        CV = CountVectorizer()
        training_frequency = CV.fit_transform(tweet)
        Tfid = TfidfTransformer()

        X_train = Tfid.fit_transform(training_frequency)
        y_train = sentiment

    ################## DRIVER ##################

    def main(self):
        df = pd.read_csv('text_emotion_sample.csv')
        content = df['content']
        sentiment = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(content, sentiment, train_size=0.8)
        
        train = self.preprocess(X_train, y_train)
        test = self.preprocess(X_test, y_test)

        models = {
            'SVC': SVC(),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'RF' : RandomForestClassifier(n_estimators=100)
        }

        # Select model here: 'SVC', 'KNN', 'RF'
        model_label = 'RF'
        model = models[model_label]

        # Select features here: 'caps', 'exclamation', 'character', 'lexicon', 'ngram', 'tfidf'
        features = ['tfidf']
        
        

        # Generate feature matrices for training and test sets
        train_feature_matrix = self.generate_feature_matrix(model_label, features, train)
        # test_feature_matrix = self.generate_feature_matrix(model_label, features, test)

        test_feature_matrix = self.generate_test_feature_matrix(model_label, features, test)
        
        # Generate output for training and test sets
        train_output = self.generate_output(train)
        test_output = self.generate_output(test)

        # Test and score
        model.fit(train_feature_matrix, train_output)
        prediction = model.predict(test_feature_matrix)
        score = f1_score(test_output, prediction, average='macro')
        print('F1 score using {} = {}'.format(model_label, score))

        # ET's stuff
        # model = self.train_embeddings(pd.read_csv('text_emotion.csv').content)
        # print(model.get_nearest_neighbors('friday'))
    
    
    def generate_test_feature_matrix(self, model_label, features, data):
        matrix = np.array([[] for _ in range(len(data))])
        for feature in features:
            label_encoder = preprocessing.LabelEncoder()
            if feature == 'lexicon':
                lexicon_matrix = np.empty((0,10))
                for tweet in data:
                    row = [x for x in tweet[2]['lexicon'].values()]
                    lexicon_matrix = np.vstack((lexicon_matrix, row))
                if model_label == 'KNN':
                    encoded_lexicon_matrix = np.empty((len(data),0))
                    for column in lexicon_matrix.T:
                        new_column = label_encoder.fit_transform(column)
                        encoded_lexicon_matrix = np.hstack((encoded_lexicon_matrix, np.transpose([new_column])))
                    lexicon_matrix = encoded_lexicon_matrix
                matrix = np.hstack((matrix, lexicon_matrix))
            elif feature == 'tfidf':
                training_frequency = self.CV.transform([tweet[0] for tweet in data])
                # TFID = TfidfTransformer(sublinear_tf=True)
                matrix = self.TFID.transform(training_frequency)
                
                # vectorizer = TfidfVectorizer()
                # feature_matrix = vectorizer.fit_transform([tweet[0] for tweet in data])
                # print(feature_matrix)
                # matrix = np.hstack((matrix, feature_matrix))
            else:
                feature_matrix = [tweet[2][feature] for tweet in data]
                if model_label == 'KNN':
                    feature_matrix = label_encoder.fit_transform(feature_matrix)
                feature_matrix = [[x] for x in feature_matrix]
                matrix = np.hstack((matrix, feature_matrix))
        print (matrix.shape)
        return matrix

    def generate_feature_matrix(self, model_label, features, data):
        matrix = np.array([[] for _ in range(len(data))])
        for feature in features:
            label_encoder = preprocessing.LabelEncoder()
            if feature == 'lexicon':
                lexicon_matrix = np.empty((0,10))
                for tweet in data:
                    row = [x for x in tweet[2]['lexicon'].values()]
                    lexicon_matrix = np.vstack((lexicon_matrix, row))
                if model_label == 'KNN':
                    encoded_lexicon_matrix = np.empty((len(data),0))
                    for column in lexicon_matrix.T:
                        new_column = label_encoder.fit_transform(column)
                        encoded_lexicon_matrix = np.hstack((encoded_lexicon_matrix, np.transpose([new_column])))
                    lexicon_matrix = encoded_lexicon_matrix
                matrix = np.hstack((matrix, lexicon_matrix))
            elif feature == 'tfidf':
                # CV = CountVectorizer()
                training_frequency = self.CV.fit_transform([tweet[0] for tweet in data])
                # TFID = TfidfTransformer(sublinear_tf=True)
                matrix = self.TFID.fit_transform(training_frequency)
                
                # vectorizer = TfidfVectorizer()
                # feature_matrix = vectorizer.fit_transform([tweet[0] for tweet in data])
                # print(feature_matrix)
                # matrix = np.hstack((matrix, feature_matrix))
            else:
                feature_matrix = [tweet[2][feature] for tweet in data]
                if model_label == 'KNN':
                    feature_matrix = label_encoder.fit_transform(feature_matrix)
                feature_matrix = [[x] for x in feature_matrix]
                matrix = np.hstack((matrix, feature_matrix))
        print (matrix.shape)
        return matrix

    def generate_output(self, data):
        return [tweet[1] for tweet in data]

if __name__ == "__main__":
    CS4248BestClass().main()
