from datetime import datetime, timedelta
import json
import uuid
import utils.logging as xartist_logging
from functools import wraps
from flask import Blueprint, jsonify, request, make_response, g

logger = xartist_logging.app_logger

authorization_api = Blueprint('authorization', __name__, url_prefix='/api/authorization')
# Concurrent Control
MAX_TOKEN_LIMIT = 1
SESSION_EXPIRE_HOURS = 2

token_cache = {}

class ASession:

    def __init__(self, token, expire_at, data):
        self.token = token
        self.expire_at = expire_at # When this token will expire
        self.created_at = datetime.now()
        self.data = data if data is not None else {}


    def get_value_map(self):
        return {'token': self.token, 'expire_at': self.expire_at}

    def get_value(self, key, default_value = None):
        v = self.data.get(key) if key in self.data else default_value
        self.data[key] = v
        return v

    def put_session_data(self, data):
        self.data = data

    def put_session_data_kv(self, key, value):
        self.data[key] = value

    def __str__(self):
        return f'ASession[Token: {self.token}, Expire at: {self.expire_at}, Created at: {self.created_at}]'

def evict_expired_session():
    if len(token_cache) >= MAX_TOKEN_LIMIT:
        # Iterate through cache to evict those expired
        current_time = datetime.now()
        expired_session = None
        for key, item in token_cache.items():
            if item.expire_at < current_time:
                # Expired
                expired_session = item
                break

        if expired_session is not None:
            token_cache.pop(expired_session.token)
            # Should have capacity now
            return True
        else:
            # All the session are active, remove failed
            return False
    else:
        # Has space for new item
        return True


@authorization_api.route('/acquire_token', methods=['GET'])
def acquire_token():
    try:
        token = request.cookies.get('token')

        asession = None
        if token is not None:
            asession = get_session(token)

        if token is None or asession is None:
            asession = _try_gen_session()

        if asession is None:
            return jsonify({'code': 'Traffic_Control', 'error_msg': 'Max token limit reached. Please try again later.'}), 407

        # Create a response object
        response = make_response(jsonify({'data': asession.get_value_map()}))
        # Set the token in the cookies if a new session was generated
        response.set_cookie('token', asession.token)
        return response
    except Exception as e:
        logger.error("Server Error", e)
        return jsonify({'code': 'Server_Error', 'error_msg': 'Server Error'}), 500




def _try_gen_session():
    logger.debug('Trying session token')
    if len(token_cache) >= MAX_TOKEN_LIMIT:
        evictSuccess = evict_expired_session()
        logger.debug(f'Evicting session token :{evictSuccess}')
        if not evictSuccess:
            logger.info('Max token limit reached')
            return None
    asession = gen_session()
    logger.info(f'Session generated :{asession}')
    token_cache[asession.token] = asession
    return asession


def gen_session():
    access_token = uuid.uuid4()
    expire_at = datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
    asession = ASession(str(access_token), expire_at, None)
    return asession


def get_session(token):
    if token in token_cache:
        return token_cache[token]
    else:
        return None


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'code': 'Token_Missing', 'error_msg': 'Token is missing'}), 401

        session = get_session(token)
        if not session:
            return jsonify({'code': 'Token_Expired', 'error_msg': 'Token is invalid or expired'}), 401

        g.asession = session  # Store the session object in the request context
        return f(*args, **kwargs)

    return decorated_function