import os
basedir = os.path.abspath(os.path.dirname(__file__))
class Config(object):
    UPLOAD_FOLDER = os.getcwd() + '/app/static/img/'