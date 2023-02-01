from flask import Flask, request
import json
import pickle
from nltk.stem import PorterStemmer

ps = PorterStemmer()
vectorizer = pickle.load(open('/home/dipadrian/mysite/vectorizer.pkl', 'rb'))


SGD05 = pickle.load(open('/home/dipadrian/mysite/SGD05.pkl', 'rb'))
logisticRegression03 = pickle.load(open('/home/dipadrian/mysite/logisticRegression03.pkl', 'rb'))
logisticRegression04 = pickle.load(open('/home/dipadrian/mysite/logisticRegression04.pkl', 'rb'))
logisticRegression05 = pickle.load(open('/home/dipadrian/mysite/logisticRegression05.pkl', 'rb'))
multinomialNB03 = pickle.load(open('/home/dipadrian/mysite/multinomialNB03.pkl', 'rb'))


app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
    r = request.get_json()
    sentences = [ps.stem(k.strip()) for k,v in r.items()]
    W = vectorizer.transform(sentences)
    severity_score = str(r['Ej7C2RIZ6VTdJt11294IbY7BZ45i2t'])

    if severity_score == "5":
            y = multinomialNB03.predict(W)
    elif severity_score == "4":
            y = logisticRegression03.predict(W)
    elif severity_score == "3":
            y = logisticRegression04.predict(W)
    elif severity_score == "2":
            y = logisticRegression05.predict(W)
    elif severity_score == "1":
            y = SGD05.predict(W)

    #sentences = [" ".join(sentence) for sentence in sentences]
    result = dict(zip(sentences, y.tolist()))
    json_dump = json.dumps(result)
    return json_dump