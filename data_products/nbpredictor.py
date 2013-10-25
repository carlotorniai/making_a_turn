import json
import pandas as pd
import numpy 	as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re	
from bs4 import BeautifulSoup
import requests



def getNYTText(url):
	'''Parses the text from a NYT page using 
	BeautifulSoup'''
	
	html = requests.get(url)
	if html.status_code == 200:
		soup = BeautifulSoup(html.text)
		for p in soup.find_all('div', class_='articleBody'):
			content = p.get_text()
	return content


def get_corpus_features(num_elements, json_input):
	''' Builds the initial features matrix 
	from the corpus contained in the json_input file'''
	data = json.load(open(json_input))	
	corpus=[]
	corpus_labels=[]
	for article in data[1:num_elements]:
	    section_title= "["+article['section_name']+"] - "+article['headline']['main']
	    if  len(article['content'])!=0:
	    	content = re.sub(r'\\xe2|\\x80|\\x99s|\\x9c|\\x9dz|\\x9d|\\x94|\\x99|xc3',\
	    	 ' ', str(article['content']))
	    	corpus.append(content)
	    	corpus_labels.append(article['section_name'])
	vectorizer = CountVectorizer(stop_words='english', min_df=1)
	Y = vectorizer.fit_transform(corpus)
	feature_matrix = Y.toarray()
   	return feature_matrix, vectorizer, corpus_labels

def build_article_feature(vectorizer, article_text):
	''' Returns the feature vector of an article'''
	content = re.sub(r'\\xe2|\\x80|\\x99s|\\x9c|\\x9dz|\\x9d|\\x94|\\x99|xc3',\
	 ' ', article_text)
	article_features =  vectorizer.transform([content]).toarray()
	return article_features[0]


def train_model(feature_matrix, corpus_labels, vectorizer, article_text=None):
	''' Trains a Multinomial Naive Bayes '''
	mnb = MultinomialNB()
	print "Shape feature Matrix" , feature_matrix.shape
	print "Labels  for training data", len(corpus_labels)
	print ("Training the model...")
	mnb.fit(feature_matrix, corpus_labels)
	return mnb
	
def predict(BNmodel, vectorizer, url, parse_text=False):
	''' Uses the trained model BNmodel to predic the section of an article.
	The flag parse_text tells if the text has to be parsed from a NYT URL.
	The vectorizer is used to build the feature vector of the new article
	according to the trained model'''

	if parse_text:
		article_content = url
	else:
		# Get text of the article first
		article_content = getNYTText(url)

	# Build the feature vector for the new article
	article_features = build_article_feature(vectorizer, article_content)
	print "Shape article featrues" , article_features.shape
	print ("Predicting....")
	article_predicted_label = BNmodel.predict(article_features)
	return article_predicted_label[0]
	