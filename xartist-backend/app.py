from flask import Flask, jsonify, request

from web.home.home import home_page
from web.api.dynamic_block import dynamic_block_api
from web.api.explorer import explorer_api
from web.api.authorization import authorization_api
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# or to allow specific origins
CORS(app, origins=["*", "http://127.0.0.1:3000"])

app.register_blueprint(home_page)
app.register_blueprint(dynamic_block_api)
app.register_blueprint(explorer_api)
app.register_blueprint(authorization_api)

@app.route('/health')
def health(name=None):
    return jsonify(
        {'Status': 'OK' }
    )