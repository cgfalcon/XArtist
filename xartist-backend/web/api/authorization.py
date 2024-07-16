from datetime import datetime, timedelta
import json
import uuid
import utils.logging as xartist_logging
from functools import wraps
from flask import Blueprint, jsonify, request, make_response, g

logger = xartist_logging.app_logger

authorization_api = Blueprint('authorization', __name__, url_prefix='/api/authorization')



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

class SessionStore:

    def __init__(self):
        # Concurrent Control
        self.MAX_TOKEN_LIMIT = 10
        self.SESSION_EXPIRE_HOURS = 24

        self.session_cache = {}


    def evict_expired_session(self):
        if len(self.session_cache) >= self.MAX_TOKEN_LIMIT:
            # Iterate through cache to evict those expired
            current_time = datetime.now()
            expired_session = None
            for key, item in self.session_cache.items():
                if item.expire_at < current_time:
                    # Expired
                    expired_session = item
                    break

            if expired_session is not None:
                self.session_cache.pop(expired_session.token)
                # Should have capacity now
                return True
            else:
                # All the session are active, remove failed
                return False
        else:
            # Has space for new item
            return True

    def _try_gen_session(self):
        logger.debug('Trying session token')
        if len(self.session_cache) >= self.MAX_TOKEN_LIMIT:
            evictSuccess = self.evict_expired_session()
            logger.debug(f'Evicting session token :{evictSuccess}')
            if not evictSuccess:
                logger.info('Max token limit reached')
                return None
        asession = self._gen_session()
        logger.info(f'Session generated :{asession}')
        self.session_cache[asession.token] = asession
        return asession

    def _gen_session(self):
        access_token = uuid.uuid4()
        expire_at = datetime.utcnow() + timedelta(hours=self.SESSION_EXPIRE_HOURS)
        asession = ASession(str(access_token), expire_at, None)
        return asession

    def get_session(self, token):
        return self.session_cache[token] if token in self.session_cache else None

    def get_or_gen_session(self, token):
        try:
            asession = None
            if token is not None and token in self.session_cache:
                asession = self.session_cache[token]

            if token is None or asession is None:
                asession = self._try_gen_session()
                self.session_cache[token] = asession

            return asession
        except Exception as e:
            logger.error("Server Error", e)
            return None


session_store = SessionStore()


@authorization_api.route('/acquire_token', methods=['GET'])
def acquire_token():
    global session_store
    try:
        token = request.cookies.get('token')

        asession = session_store.get_or_gen_session(token)

        if asession is None:
            return jsonify({'success': False, 'code': 'Traffic_Control', 'error_msg': 'Max token limit reached. Please try again later.'}), 200

        # Create a response object
        response = make_response(jsonify({'success': True, 'data': asession.get_value_map()}))
        # Set the token in the cookies if a new session was generated
        response.set_cookie('token', asession.token)
        return response
    except Exception as e:
        logger.error("Server Error", e)
        return jsonify({'success': False, 'code': 'Server_Error', 'error_msg': 'Server Error'}), 200



def token_required(f):
    global session_store
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global session_store
        token = request.headers.get('Authorization', None)
        if not token:
            return jsonify({'success': False, 'code': 'Token_Missing', 'error_msg': 'Token is missing'}), 401

        if token not in session_store.session_cache:
            return jsonify({'success': False, 'code': 'Token_Expired', 'error_msg': 'Token is invalid or expired'}), 401

        g.asession = session_store.session_cache[token]  # Store the session object in the request context
        return f(*args, **kwargs)

    return decorated_function