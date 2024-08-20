import logging
from logging.config import dictConfig
import contextvars

task_id_var = contextvars.ContextVar('task_id', default='null')

class RequestFilter(logging.Filter):
    def filter(self, record):
        record.task_id = task_id_var.get()
        return True

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] [%(levelname)s] [%(task_id)s] %(message)s',
        },
    },
    'filters': {
        'request_filter': {
            '()': RequestFilter,
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'filters': ['request_filter'],
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
}

dictConfig(logging_config)

logger = logging.getLogger()

def set_request_context(task_id):
    task_id_var.set(task_id)

def get_task_id():
    return task_id_var.get()