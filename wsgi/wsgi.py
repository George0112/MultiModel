import os

from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html', name="index")

@app.route('/submit', methods=['POST'])
def submit():
    os.system("argo submit ../multimodel.yaml")
    return 'ok'
