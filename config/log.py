import os
import logging
from pytz import timezone
from datetime import datetime
from logging.config import dictConfig
from config.config import LOG_LEVEL

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.makedirs('./logs', exist_ok=True)

def timetz(*args):
    tz = timezone('Asia/Kolkata')
    return datetime.now(tz).timetuple()

LOGGING = {
    'version':1,
    'disable_existing_loggers':False,
    'formatters':{
        'standard':{
            'class':'logging.Formatter',
            'format':'%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers':{
        'console': {
            'class': 'logging.StreamHandler',
            'level': f'{LOG_LEVEL}',
            'formatter': 'standard',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024*1024*1, # 1 MB,
            'backupCount': 1,
            'level': f'{LOG_LEVEL}',
            'formatter': 'standard',
            'filename': 'logs/repo.log'
        }
    },
    'loggers':{
        'repo_controller':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'repo':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'upload_controller':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'upload':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'preprocessor':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'embedder':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        },
        'similarity':{
            'handlers': ['file','console'],
            'level':f'{LOG_LEVEL}',
            'propagate':True
        }
    }
}

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] [%s] %s' % (self.extra['executionId'], self.extra['project'], msg), kwargs

class CustomAdapterUpload(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['project'], msg), kwargs
    
logging.Formatter.converter = timetz
dictConfig(LOGGING)
repo_cnt_logger = logging.getLogger('repo_controller')
repo_logger = logging.getLogger('repo')
upload_cnt_logger = logging.getLogger('upload_controller')
upload_logger = logging.getLogger('upload')
preprocessor_logger = logging.getLogger('preprocessor')
embedder_logger = logging.getLogger('embedder')
similarity_logger = logging.getLogger('similarity')