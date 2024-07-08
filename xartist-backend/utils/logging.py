import logging
from logging.config import dictConfig
import json

from .project_configs_helper import ProjectConfig

# Default logger settings
default_logger_level = 'INFO'
default_format = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >> %(message)s"
default_app_logger_file = 'app.log'
default_error_logger_file = 'error.log'
default_flask_logger_file = 'flask.log'

app_logger_name = "app_logger"
error_logger_name = "error_logger"
flask_logger_name = 'flask_logger'

if 'logger' not in ProjectConfig:
    # Set default logger
    print(f'Logger configuration missing. Logger will be inited with default settings.')
    ProjectConfig['logger'] = {
        app_logger_name: {
            "level": default_logger_level,
            "format": default_format,
            "filename": default_app_logger_file},
        error_logger_name: {
            "level": default_logger_level,
            "format": default_format,
            "filename": default_error_logger_file}
    }


# flask_logger
flask_logger_config = ProjectConfig['logger'][flask_logger_name]

# app_logger
app_logger_config = ProjectConfig['logger'][app_logger_name]

# exception_logger
error_logger_config = ProjectConfig['logger'][error_logger_name]

# Init logging configuration by dictConfig
dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": default_format,
            },
            "app_logger_formatter": {
                "format": app_logger_config['format'] if app_logger_config['format'] is not None else default_format
            },
            "exception_logger_formatter": {
                "format": error_logger_config['format'] if error_logger_config['format'] is not None else default_format
            },
            "flask_logger_formatter": {
                "format": flask_logger_config['format'] if flask_logger_config['format'] is not None else default_format
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "app_log_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": app_logger_config['filename'] if app_logger_config['filename'] is not None else default_app_logger_file,
                "maxBytes": 500000000, # 500MB
                "backupCount": 5,
                "formatter": "app_logger_formatter",
            },
            "exception_log_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": error_logger_config['filename'] if error_logger_config['filename'] is not None else default_error_logger_file,
                "maxBytes": 500000000, # 500MB
                "backupCount": 5,
                "formatter": "exception_logger_formatter",
            },
            "flask_log_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": flask_logger_config['filename'] if flask_logger_config['filename'] is not None else default_flask_logger_file,
                "maxBytes": 500000000, # 500MB
                "backupCount": 5,
                "formatter": "flask_logger_formatter",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "flask_log_handler"]},
        "loggers": {
            "app_logger": {
                "level": app_logger_config['level'] if app_logger_config['level'] is not None else "INFO",
                "handlers": ["app_log_handler"],
                "propagate": True,
            },
            "exception_logger": {
                "level": error_logger_config['level'] if error_logger_config['level'] is not None else "ERROR",
                "handlers": ["exception_log_handler"],
                "propagate": True,
            }
        }
    }
)


def log_function(api_function):
    """
    API Logger decorator. By using this decorator, the parameters and result of the invoked function will be recorded
    :param api_function: target function
    :return: a decorated function
    """
    def inner(*args, **kwargs):
        try:
            ret = api_function(*args, **kwargs)
            log_message = json.dumps({
                "status": "Success",
                "api_name": api_function,
                "args": args,
                "kwargs": kwargs,
                "response": ret
            })
            app_logger.info(log_message)
        except Exception as e:
            error_message = {
                "status": "Failed",
                "api_name": api_function,
                "args": args,
                "kwargs": kwargs,
                "response": "API Exception"
            }
            app_logger.info(json.dumps(error_message))

            del error_message['status']
            error_message['exception'] = str(e)
            error_logger.error(json.dumps(error_message))
            raise
        return ret
    return inner


# Init logger
app_logger = logging.getLogger('app_logger')
error_logger = logging.getLogger('error_logger')