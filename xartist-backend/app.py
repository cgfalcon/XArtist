from flask import Flask, jsonify, request


app = Flask(__name__)


@app.route('/health')
def health(name=None):
    return jsonify(
        {'Status': 'OK' }
    )