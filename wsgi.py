import flask
from flask import request, jsonify, render_template
import pandas as pd
import sys
import traceback
import os

from metatagger import findMatches

app = flask.Flask(__name__)
app.config["DEBUG"] = True


wiki_data = pd.read_csv("workingdata.csv")


@app.route('/api/v1/metatagger/health', methods=['GET'])
def health():
    result = {
        "status": "UP"
    }
    return jsonify(result)


@app.route('/api/v1/metatagger/refdata', methods=['GET'])
def refDataEndPoint():
    return wiki_data.to_json()


@app.route('/api/v1/metatagger/predict', methods=['GET'])
def predictCategory():
    result = {
        "inputtext": "sometext",
        "result": [
            {
                "category": "basketball",
                "matchedpercentage": "70"
            }
        ]
    }
    return jsonify(result)


@app.route('/api/v1/metatagger/predict', methods=['POST'])
def handlePredictPost():
    try:
        content = request.json
        input_query = content['query']
        result = findMatches(input_query)
        data = {
            "data": input_query,
            "result": result,
        }
        return jsonify(data)
    except:
        print("Whew!", sys.exc_info(), "occured")
        err = traceback.format_exc()
        return jsonify({
            "data": "some error",
            "result": err
        }), 500


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/pages/query")
def queryhomehtml():
    return render_template('queryhome.html')


@app.route("/pages/referancedata")
def referancedatahtml():
    return render_template('referancedata.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
