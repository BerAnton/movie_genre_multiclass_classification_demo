#!/usr/bin/env python3

"""Make genres prediction on given dialogue from movie"""

sys.path.append("../app")

import os
import sys
from flask import Flask, render_template, request

from train import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH
from classifier import Classifier

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index_page(text="", prediction_message=""):
    """Main page of web app"""
    if request.method == "POST":
        clf = Classifier(DEFAULT_MODEL_PATH,
                         DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH)
        dialogue = request.form["text"]
        prediction = clf.predict(dialogue)
        prediction_message = " ".join(sorted(prediction))
    return render_template('prediction_page.html', text=text, prediction_message=prediction_message)
