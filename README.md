# Notion of Twitter Emotion

This is a CS4248 Project done by Team 15. 

## Introduction

Every second, an average of 6,000 tweets are posted on Twitter, with many indicating some form of emotion. With these tweets, we hope to accurately determine the emotions associated with it using NLP techniques and be able to generalize the emotional sentiments attached to different issues. Using this, we could then build a sentiment report that is useful to different researchers in learning about their topics. With that in mind, we implemented our own preprocessing methods and utilised **SVM**, **Multinomial Naive Bayes**, **Random Forest** and **k-NN** classifiers to build and train our model.

## Quick Start
To start our model, the following pre-requisites are needed.

### Pre-requisites
* [Python](https://www.python.org/downloads/)
* [Github](https://github.com/)

First, clone our repository by running this command:
```
git clone https://github.com/grrrrnt/notion-emotion-twitter.git
```

Second, download the respective python libraries by running this command in the root directory:

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
