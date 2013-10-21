import json
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
import sys 
import re

def get_corpus_features(num_elements, source_file):
	''' Parse the json file with the conent of the NYT artiles
	builds the feature matrix and returns it together with the 
	lists of articles and the vectorizer for the analysis of results '''

	data = json.load(open(source_file))	
	corpus=[]
	corpus_titles=[]
	# Just using a subset of the data with just 100 articles
	for article in data[1:num_elements]:
	    section_title= "["+article['section_name']+"] - "+article['headline']['main']
	    if  len(article['content'])!=0:
	    	# Get the text stripe some odd chars and add it to the corpus
	    	content = re.sub(r'\\xe2|\\x80|\\x99s|\\x9c|\\x9dz|\\x9d|\\x94|\\x99|xc3', ' ', str(article['content']))
	    	corpus.append(content)
	    	corpus_titles.append(section_title)
	vectorizer = CountVectorizer(stop_words='english', min_df=1)
	Y = vectorizer.fit_transform(corpus)
	feature_matrix = Y.toarray()
   	return feature_matrix, corpus_titles, vectorizer

def cost(f_vectors, w_matrix, h_matrix):
	''' Computes the cost funciton as the mean square error between 
	the original feature matrix and the approximated matrix H dot V '''
	# Caclualte the HW matrix
	v_approx = np.dot(np.array(w_matrix), np.array(h_matrix))
	# Get the difference between the actual features and the approx features
	diff_matrix = f_vectors - v_approx
	cost_value = np.sum(np.array(diff_matrix)**2)
	return cost_value

def update_h(f_vectors, h_matrix, w_matrix,):
	''' Computes the new H matrix '''
	w_matrix_T = np.transpose(w_matrix)
	h_matrix_T = np.transpose(h_matrix)
	new_h_matrix = (np.matrix(w_matrix_T) * np.matrix(f_vectors) ) / ( np.matrix(w_matrix_T) * np.matrix(w_matrix) * np.matrix(h_matrix) )
	final_h_matrix = np.array(new_h_matrix) * np.array(h_matrix)
	return final_h_matrix

def update_w(f_vectors, h_matrix, w_matrix,):
	''' Computes the new W matrix'''
	w_matrix_T = np.transpose(w_matrix)
	h_matrix_T = np.transpose(h_matrix)
	new_w_matrix = (np.matrix(f_vectors) * np.matrix(h_matrix_T) ) / ( np.matrix(w_matrix) * np.matrix(h_matrix) * np.matrix(h_matrix_T) )
	final_w_matrix = np.array(new_w_matrix) * np.array(w_matrix)
	return final_w_matrix	

def get_summary(f_vectors, h_matrix, w_matrix, vectorizer, corpus_titles):
	''' Returns the top 10 words and the top 5 Articles for each feature '''
	
	total_top_words=dict()
	total_top_articles=dict()
	
	# For each Row in the H_Matrix i.e. for each feature
	for feature_num in range(h_matrix.shape[0]):
		print "processing feature %d..." %(feature_num)
		sorted_index_h_matrix = np.argsort(h_matrix[feature_num])
		max_index = sorted_index_h_matrix[-10:]
		top_words=[]
		words = vectorizer.get_feature_names()
		for index in max_index:
		    top_words.append(words[index])
		
		# Store the words in teh dictionary
		total_top_words[feature_num] = top_words
		f1_weigths =[]
		
		# Get the 5 top articles
		for i in range(len(w_matrix)):
			f1_weigths.append(w_matrix[i][feature_num])
			sorted_f1_index = np.argsort(f1_weigths)
    		max_art_index = sorted_f1_index[-5:]
    		top_articles=[]
    		for art_index in max_art_index:
    			top_articles.append(corpus_titles[art_index])
    		total_top_articles[feature_num] = top_articles
	return total_top_words, total_top_articles

def compute_nmf (f_vectors, r, max_iter):
	''' Computes the nmf given the feature vector,
	the number of features and the max number of iteration
	'''

	iterations = 0;
	convergence = False
	n = f_vectors.shape[0]
	m = f_vectors.shape[1]

	# As first step initialize H and W with random values
	w_matrix = np.random.random((n, r))
	h_matrix = np.random.random((r, m))

	print "Sanity check:"
	print "V matrix: " , f_vectors.shape
	print "W matrix: " , w_matrix.shape
	print "H matrix: " , h_matrix.shape

	# Run a first update without calculating the cost 
	h_matrix = update_h(f_vectors, h_matrix, w_matrix)
	w_matrix = update_w(f_vectors, h_matrix, w_matrix)

	# Initialize this list to keep track of cost
	cost_series = []
	while not convergence and iterations < max_iter:

		# Compute the cost function
		current_cost = cost(f_vectors, w_matrix, h_matrix)
		cost_series.append(current_cost)
		if current_cost==0:
			convergence = True
			return w_matrix, h_matrix, cost_series
			break
		else:
			# Update H and W
			h_matrix = update_h(f_vectors, h_matrix, w_matrix)
			w_matrix = update_w(f_vectors, h_matrix, w_matrix)
		iterations+=1
	return w_matrix, h_matrix, cost_series