import nbpredictor
import pickle


# Build the corpus and compute related features
feature_matrix, vectorizer, corpus_labels = nbpredictor.get_corpus_features(900, './data/articles_html1000.json')
print corpus_labels, len(corpus_labels)


# Build the trained model
trained_model =  nbpredictor.train_model(feature_matrix, corpus_labels, vectorizer)

# Test the classification for the text
print nbpredictor.predict(trained_model, vectorizer, 'Obama Syria UN', True)

# Test the classificaiton from an URL
print nbpredictor.predict(trained_model, vectorizer, 'http://www.nytimes.com/2013/10/17/us/congress-budget-debate.html?hp')

# Save the trained model
outfile_model = open("trained_nb_model.pkl", "wb")
pickle.dump(trained_model, outfile_model)
outfile_model.close()

# Save the vectorizer
outfile_vectorizer = open("vectorizer.pkl", "wb")
pickle.dump(vectorizer, outfile_vectorizer)
outfile_vectorizer.close()

