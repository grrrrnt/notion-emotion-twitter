# Notion of Twitter Emotion

This is a CS4248 Project done by Team 15. 

## Introduction

Every second, an average of 6,000 tweets are posted on Twitter, with many indicating some form of emotion. With these tweets, we hope to accurately determine the emotions embedded within them using natural language processing (NLP) techniques and be able to generalise the emotional sentiments attached to different topics. Using this model, we aim to design a tool that generates sentiment reports which could prove to be useful in the work of researchers. With that in mind, we conceptualised and implemented our own preprocessing methods and compared between **Support Vector Machine**, **Multinomial Naive Bayes**, **Random Forest** and **k-Nearest Neighbours** classifiers to build, train and tune our model.

## Project Report

Please refer to our [project report](../main/Group%2015%20-%20Project%20Final%20Report.pdf) for the results of our analysis.

## Quick Start
To start our model, the following pre-requisites are needed:

### Pre-requisites
* [Python](https://www.python.org/downloads/)
* [GitHub](https://github.com/)

First, clone our repository by running this command:
```
git clone https://github.com/grrrrnt/notion-emotion-twitter.git
```

Second, download the respective libraries by running this command in the root directory:

```
pip3 install -r requirement.txt
```

Lastly, with the data file in _text_emotion.csv_, run our models with this command:

```
python3 notion_emotion_twitter.py
```

* To compare the performance between models, uncomment `line 384` of  _notion_emotion_twitter.py_ and comment out `line 385` instead.
* To modify the model that is being run, refer to `line 234` of _notion_emotion_twitter.py_ and change it accordingly.
* To select the different features, refer to `line 238` of _notion_emotion_twitter.py_ and change it accordingly.
* To test the RF model on unseen dataset, uncomment `line 385` of  _notion_emotion_twitter.py_ and comment out `line 384` instead.
* To change the unseen dataset, refer to `line 355` of _notion_emotion_twitter.py_ and change it accordingly.

## Datasets
The datasets that we have used have been obtained from Kaggle:
* [Emotion tweets (Training dataset)](https://www.python.org/downloads/)
* [Tweets about Donald Trump](https://www.kaggle.com/gpreda/trump-tweets)
* [Tweets about Elon Musk](https://www.kaggle.com/kulgen/elon-musks-tweets)
* [Tweets about COVID-19](https://www.kaggle.com/gpreda/covid19-tweets)
* [Tweets about COVID-19 vaccination](https://www.kaggle.com/gpreda/pfizer-vaccine-tweets)
