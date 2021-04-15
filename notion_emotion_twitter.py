import os
import re
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class TwitterEmotion:
    PROGRESS = True
    USE_CACHE = True
    ABBREV_CSV = pd.read_csv('abbreviations.csv')
    ABBREV_DICT = dict(zip(ABBREV_CSV.abbreviation, ABBREV_CSV.replacement))
    WORD_PATTERN = re.compile('\w+')
    STOPWORDS = stopwords.words('english')
    PRETRAINED_MODEL_PATH = 'lid.176.ftz'
    LANGUAGE_MODEL = fasttext.load_model(PRETRAINED_MODEL_PATH)
    SPELL = SpellChecker(distance=1)
    CV = CountVectorizer(ngram_range=[1,1])
    TFID = TfidfTransformer(sublinear_tf=True)
    TFID_CV = [CountVectorizer(ngram_range=r) for r in [[1,3], [1,1], [2,2], [3,3], [1,2]]]

    ################## PREPROCESSING ##################

    def clean(self, df):
        df = df.drop(df[df['sentiment'] == 'empty'].index)
        df = df.drop(df[df['sentiment'] == 'neutral'].index)
        df = df.drop(df[df['sentiment'] == 'boredom'].index)
        df['sentiment'].replace(to_replace='hate', value='anger', inplace=True)
        df['sentiment'].replace(to_replace=['love','fun','relief','enthusiasm'], value='happiness', inplace=True)
        print(df.groupby('sentiment')['sentiment'].count().sort_values(ascending=False))
        return df

    def preprocess(self, X, y):
        data = list(zip(X, y))
        preprocessed = []
        for tweet, sentiment in tqdm(data, 'Preprocess', disable=not self.PROGRESS):
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
        removed_http = re.sub(r'https?://[a-zA-Z0-9_./]+', ' ', tweet)
        return re.sub(r'www\.[a-zA-Z0-9_./]+', ' ', removed_http)

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

    def train_embeddings(self, train, supervised):
        content_file_name = 'text_content.txt'
        if supervised:
            with open(content_file_name, 'w') as f:
                f.writelines(tweet + ' __label__' + sentiment + '\n' for tweet, sentiment, _ in train)
            model = fasttext.train_supervised(content_file_name)
        else:
            with open(content_file_name, 'w') as f:
                f.writelines(tweet + '\n' for tweet, _, _ in train)
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

    ################## DRIVER ##################

    def main(self):
        df = pd.read_csv('text_emotion.csv')
        df = self.clean(df)
        content = df['content']
        sentiment = df['sentiment']

        # Use this line for a randomised train-test split (tuning model parameters)
        # X_train, X_test, y_train, y_test = train_test_split(content, sentiment, train_size=0.8)
        
        # Use this line for a fixed train-test split (comparing performance between models)
        X_train, X_test, y_train, y_test = self.interval_train_test_split(content, sentiment)

        train = self.preprocess(X_train, y_train)
        test = self.preprocess(X_test, y_test)

        models = {
            'SVC': SVC(kernel='sigmoid'),
            'KNN': KNeighborsClassifier(n_neighbors=1),
            'RF' : RandomForestClassifier(n_estimators=100),
            'MNB': MultinomialNB()
        }

        # Select model here: 'SVC', 'KNN', 'RF', 'MNB'
        model_label = 'SVC'
        model = models[model_label]

        # Select features here: 'caps', 'exclamation', 'character', 'lexicon', 'tfidf[0-4]', 'embed', 'count'
        feature_combinations = [['count'], ['embed'], ['lexicon'], ['caps'], ['exclamation'], ['character']]

        for features in feature_combinations:
            # Generate feature matrices for training and test sets
            train_feature_matrix = self.generate_feature_matrix(model_label, features, train, False)
            test_feature_matrix = self.generate_feature_matrix(model_label, features, test, True)
            
            # Generate output for training and test sets
            train_output = self.generate_output(train)
            test_output = self.generate_output(test)

            if self.PROGRESS:
                print(f'Train {features}')
            # Test and score
            model.fit(train_feature_matrix, train_output)
            prediction = model.predict(test_feature_matrix)
            score = f1_score(test_output, prediction, average='macro')
            print('F1 score using {} with {} feature = {}'.format(model_label, features, score))

    TRAIN_CACHE = {}
    TEST_CACHE = {}

    def generate_feature_matrix(self, model_label, features, data, is_test):
        if self.USE_CACHE:
            if is_test:
                cache = self.TEST_CACHE
            else:
                cache = self.TRAIN_CACHE
            for feature in features:
                if feature not in cache:
                    cache[feature] = self.generate_feature(model_label, feature, data, is_test)
            return np.hstack(list(cache[feature] for feature in features))
        else:
            matrix = np.empty((len(data), 0))
            for feature in features:
                feature_values = self.generate_feature(model_label, feature, data, is_test)
                matrix = np.hstack((matrix, feature_values))
            return matrix

    def generate_feature(self, model_label, feature, data, is_test):
        if self.PROGRESS:
            print(f'Generate {feature}, is_test={is_test}')
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
            return lexicon_matrix
        elif feature.startswith('tfidf'):
            cv = self.TFID_CV[int(feature[5])]
            if is_test:
                training_frequency = cv.transform([tweet[0] for tweet in data])
                return self.TFID.transform(training_frequency).todense()
            else:
                training_frequency = cv.fit_transform([tweet[0] for tweet in data])
                return self.TFID.fit_transform(training_frequency).todense()
        elif feature == 'count':
            if is_test:
                return self.CV.transform([tweet[0] for tweet in data]).todense()
            else:
                return self.CV.fit_transform([tweet[0] for tweet in data]).todense()
        elif feature == 'embed':
            if not is_test:
                self.embed_model = self.train_embeddings(data, supervised=True)
                print(self.embed_model.get_nearest_neighbors('friday'))
            embed = [self.embed_model.get_sentence_vector(tweet) for tweet, _, _ in data]
            if model_label == 'MNB':
                if is_test:
                    embed = self.embed_scaler.transform(embed)
                else:
                    self.embed_scaler = preprocessing.MinMaxScaler((0, 10))
                    embed = self.embed_scaler.fit_transform(embed)
            return embed
        else:
            feature_matrix = [tweet[2][feature] for tweet in data]
            if model_label == 'KNN':
                feature_matrix = label_encoder.fit_transform(feature_matrix)
            return [[x] for x in feature_matrix]

    def generate_output(self, data):
        return [tweet[1] for tweet in data]

    def interval_train_test_split(self, X, y):
        X_train = X.iloc[X.index % 5 != 0]
        y_train = y.iloc[y.index % 5 != 0]
        X_test = X.iloc[X.index % 5 == 0]
        y_test = y.iloc[y.index % 5 == 0]
        return X_train, X_test, y_train, y_test

    ################## FOR QUESTION 2 AND 3 ##################
    # 1. Use best model to train on the entire train dataset
    # 2. Get separate unseen dataset (Covid-19 dataset)
    # 3. Get frequency count of tokens (keywords)
    # 4. Output top 20 words
    # 5. Display pie chart of emotion predictions

    def test_on_unseen_dataset(self):
        model_label = 'RF'                                      # to update with best performing params
        model = RandomForestClassifier(n_estimators=100)
        features = ['count', 'embed', 'lexicon']
        
        train_df = pd.read_csv('text_emotion.csv')
        train_df = self.clean(train_df)
        train_content = train_df['content']
        train_sentiment = train_df['sentiment']
        train = self.preprocess(train_content, train_sentiment)
        train_feature_matrix = self.generate_feature_matrix(model_label, features, train, False)
        
        # Select topical dataset:
        # covid19_tweets(_sample).csv, vaccination_tweets(_sample).csv, trump_tweets.csv, elonmusk_tweets.csv
        test_df = pd.read_csv('unseen_datasets/elonmusk_tweets.csv')
        test_content = test_df['text']
        test = self.preprocess(test_content, np.empty((test_df.shape[0],)))
        test_feature_matrix = self.generate_feature_matrix(model_label, features, test, True)

        train_output = self.generate_output(train)
        model.fit(train_feature_matrix, train_output)
        prediction = model.predict(test_feature_matrix)

        test_tweets = [x[0] for x in test]
        output = pd.DataFrame({'tweet': test_tweets, 'prediction': prediction}, columns=['tweet', 'prediction'])
        output.to_csv('output.csv', mode='w+')

        sentiments = train_sentiment.unique()
        most_frequent_tokens = {}
        for sentiment in sentiments:
            new_df = output[output['prediction'] == sentiment]
            count = dict(pd.Series(' '.join(new_df['tweet']).split()).value_counts()[:20])
            most_frequent_tokens[sentiment] = count
        
        # Print 20 most frequent tokens for each emotion
        print(most_frequent_tokens)
        
        # Display pie chart of emotions
        chart = output['prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        chart.get_figure().savefig('output')
        plt.show()

if __name__ == "__main__":
    # TwitterEmotion().main()
    TwitterEmotion().test_on_unseen_dataset()
