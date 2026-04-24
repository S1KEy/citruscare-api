# application.py - AWS Elastic Beanstalk entry point
from api import app as application

if __name__ == '__main__':
    application.run()