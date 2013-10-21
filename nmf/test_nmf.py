import nmf

feature_matrix, corpus_titles, vectorizer = nmf.get_corpus_features(100, "./data/articles_html1000.json")
print feature_matrix.shape

# Test nmf with 4 features and 100 max iteration
w_matrix, h_matrix, cost_series = nmf.compute_nmf(feature_matrix, 4, 100)


	