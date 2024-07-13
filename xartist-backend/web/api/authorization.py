from datetime import datetime
import utils.logging as xartist_logging
from flask import Blueprint, jsonify, request

logger = xartist_logging.app_logger

authorization_api = Blueprint('authorization', __name__, url_prefix='/api/authorization')
# Concurrent Control
max_token_limit = 5

token_cache = {}

class SessionToken:

    def __init__(self, token, expire_at, data):
        self.token = token
        self.expire_at = expire_at # When this token will expire
        self.created_at = datetime.now()
        self.data = data # Session Storage

    def get_value_map(self):
        return {'token': self.token, 'expire_at': self.expire_at}


def evict_expired_session():
    if len(token_cache) >= max_token_limit:
        # Iterate through cache to evict those expired
        current_time = datetime.now()
        expired_session = None
        for it in token_cache.items():
            if it.expire_at < current_time:
                # Expired
                expired_token = it
                break

        if expired_session:
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
    token = request.cookies['token']

    asession = _try_gen_session()
    if asession is None:
        return jsonify({'error': 'Max token limit reached. Please try again later.'}), 407

    return jsonify({'data': asession})


def _try_gen_session():
    logger.debug('Trying session token')
    if len(token_cache) >= max_token_limit:
        evictSuccess = evict_expired_session()
        logger.debug(f'Evicting session token :{evictSuccess}')
        if not evictSuccess:
            logger.info('Max token limit reached')
            return None
    asession = gen_session()
    logger.into(f'Session generated :{asession}')
    token_cache[asession.token] = asession.get_value_map()
    return asession


def gen_session():
    access_token = uuid.uuid4()
    access_token_obj = {
        'access_token': str(access_token),
        'timestamp': time.time(),
    }
    return access_token, access_token_obj


def get_session(token):
    if token in token_cache:
        return token_cache[token]
    else:
        # Session Expired
        #   Try to acquire new token
        asession = _try_gen_session()
        if asession is None:
            raise ValueError('Session Expired')