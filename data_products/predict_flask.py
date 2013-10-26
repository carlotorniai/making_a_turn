import json
from flask import Flask, request, Response, redirect, url_for
import pickle
import pdb
from nbpredictor import predict

app = Flask(__name__)


def readpickle(filename):
    ''' Reads a pickle file and returns
     its content'''
    infile = open(filename, "rb")
    content = pickle.load(infile)
    infile.close()
    return content

# Load the trained model and the vectorizer
trained_model = readpickle('trained_nb_model.pkl')
print type(trained_model)
vectorizer = readpickle('vectorizer.pkl')
print type(vectorizer)

@app.route("/parsetext", methods=['POST'])
def execute_text():
    text = request.form['text']
    print text
    if request.method == 'POST':
        if text=='':
            results = "You should post some content!"
        else:
            label = predict(trained_model, vectorizer, text, True)
            print label
            results = "Your text belongs to the " + label + " section of the NYT"
        return Response(results, status=200,  mimetype='text/plain')
    else:
        return "Post a URL form a NYT article!!"

# Order of routes matters
@app.route("/<name>")
def hello(name):
    return "Hello " + name + "!\nWelcome to my NYT article section Predictor! )"

@app.route("/")
def index():
    return redirect(url_for('static', filename='index.html'))
   

if __name__ == "__main__":
    app.run(host='0.0.0.0')