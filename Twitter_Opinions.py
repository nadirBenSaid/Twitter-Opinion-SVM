
 # Copyright (C) 2019 Bensaid Nadir (Bensaid.nadir@gmail.com)

 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.

 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.

 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import re, base64, tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


Training_mode = False
if Training_mode:
	data = pd.read_csv("raw_data.csv")
	nltk.download('wordnet')
	nltk.download('punkt')
	nltk.download('stopwords')
	data_clean = data.copy()
	data_clean['sentiment'] = data_clean[base64.b64decode("YWlybGluZQ==")+'_sentiment'].\
		apply(lambda x: 1 if x=='negative' else 0)
	data_clean['text_clean'] = data_clean['text'].apply(lambda x: BeautifulSoup(x, "lxml").text)
	data_clean['sentiment'] = data_clean[base64.b64decode("YWlybGluZQ==")+'_sentiment'].apply(lambda x: 1 if x=='negative' else 0)
	data_clean = data_clean.loc[:, ['text_clean', 'sentiment']]
	data_clean.head()

	train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
	X_train = train['text_clean'].values
	X_test = test['text_clean'].values
	y_train = train['sentiment']
	y_test = test['sentiment']

def tokenize(text): 
	tknzr = TweetTokenizer()
	return tknzr.tokenize(text)

def stem(doc):
	return (stemmer.stem(w) for w in analyzer(doc))

if Training_mode:
	global grid_svm
	en_stopwords = set(stopwords.words("english")) 
	vectorizer = CountVectorizer(
		analyzer = 'word',
		tokenizer = tokenize,
		lowercase = True,
		ngram_range=(1, 1),
		stop_words = en_stopwords)
	kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
	np.random.seed(1)
	pipeline_svm = make_pipeline(vectorizer, 
								SVC(probability=True, kernel="linear", class_weight="balanced"))
	grid_svm = GridSearchCV(pipeline_svm,
						param_grid = {'svc__C': [0.01, 0.1, 1]}, 
						cv = kfolds,
						scoring="roc_auc",
						verbose=1,   
						n_jobs=-1) 
	grid_svm.fit(X_train, y_train)
	grid_svm.score(X_test, y_test)
	grid_svm.best_params_
	grid_svm.best_score_

def report_results(model, X, y):
	pred_proba = model.predict_proba(X)[:, 1]
	pred = model.predict(X)		

	auc = roc_auc_score(y, pred_proba)
	acc = accuracy_score(y, pred)
	f1 = f1_score(y, pred)
	prec = precision_score(y, pred)
	rec = recall_score(y, pred)
	result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
	return result

if Training_mode:
	report_results(grid_svm.best_estimator_, X_test, y_test)
def get_roc_curve(model, X, y):
	pred_proba = model.predict_proba(X)[:, 1]
	fpr, tpr, _ = roc_curve(y, pred_proba)
	return fpr, tpr

if Training_mode:
	roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
	fpr, tpr = roc_svm

if Training_mode:
	plt.figure(figsize=(14,8))
	plt.plot(fpr, tpr, color="red")
	plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Roc curve')
	plt.show()


from sklearn.model_selection import learning_curve

if Training_mode:
	train_sizes, train_scores, test_scores = \
		learning_curve(grid_svm.best_estimator_, X_train, y_train, cv=5, n_jobs=-1, 
					   scoring="roc_auc", train_sizes=np.linspace(.1, 1.0, 10), random_state=1)

def plot_learning_curve(X, y, train_sizes, train_scores, test_scores, title='', ylim=None, figsize=(14,8)):

	plt.figure(figsize=figsize)
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="lower right")
	return plt

if Training_mode:
	plot_learning_curve(X_train, y_train, train_sizes, train_scores, test_scores, ylim=(0.7, 1.01), figsize=(14,6))
	plt.show()
	joblib.dump(grid_svm, 'svmTrained.pkl', compress=3)
	print grid_svm.predict(["I liked it @nadir"])
else:
	classifier = joblib.load('svmTrained.pkl')



class TwitterAPI(object):

	def __init__(self):
		consumer_key = ''			#
		consumer_secret = ''		# fill up with your Twitter App credentials
		access_token = ''			#
		access_token_secret = ''	#
		try:
			self.auth = OAuthHandler(consumer_key, consumer_secret)
			self.auth.set_access_token(access_token, access_token_secret)
			self.api = tweepy.API(self.auth)
		except:
			print("Error: Authentication Failed")
 
	def clean_tweet(self, tweet):
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

	def get_tweet_sentiment(self, tweet):
		analysis = TextBlob(self.clean_tweet(tweet))
		if analysis.sentiment.polarity > 0:
			return 'positive'
		elif analysis.sentiment.polarity == 0:
			return 'neutral'
		else:
			return 'negative'
 
	def get_tweets(self, query, count = 10):
		tweets = []
		try:
			fetched_tweets = self.api.search(q = query, count = count)
			for tweet in fetched_tweets:
				parsed_tweet = {}
 
				parsed_tweet['text'] = tweet.text
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
 
				if tweet.retweet_count > 0:
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)
 
			return tweets
 
		except tweepy.TweepError as e:
			print("Error : " + str(e))
 
def main():
	api = TwitterAPI()
	tweets = api.get_tweets(query = '', count = 200) # Keyword to look for
	classifier = joblib.load('svmTrained.pkl')
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
	print("Neutral tweets percentage: {} % \
		".format(100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))
 
	print("\n\nPositive tweets:")
	for tweet in ptweets[:10]:
		print(tweet['text'])
 
	print("\n\nNegative tweets:")
	for tweet in ntweets[:10]:
		print(tweet['text'])
 
if __name__ == "__main__":
	main()
