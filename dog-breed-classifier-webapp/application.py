from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)
from classifier import Classifier


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    app.logger.debug('Uploaded {}'.format(f))

    prediction = Classifier.classify(f)

    return render_template('result.html', prediction=prediction, filename=f.filename)
