import traceback
import sys
import datetime

class Logger(object):
    def __init__(self):
        pass

    def log(self,log_string):
        sys.stdout.write('LOG (%s) %s\n' % (datetime.datetime.utcnow(),log_string))
        sys.stdout.flush()
        return 1

    def warning(self,log_string):
        sys.stdout.write('WARNING (%s) %s\n' % (datetime.datetime.utcnow(),log_string))
        sys.stdout.flush()
        return 1

    def error(self,error_string=''):
        sys.stdout.write('ERROR (%s) %s\n' % (datetime.datetime.utcnow(),error_string))
        sys.stdout.write('%s\n' % sys.exc_info()[0])
        sys.stdout.write('%s\n' % sys.exc_info()[1])
        sys.stdout.write('%s\n' % traceback.print_tb(sys.exc_info()[2]))
        sys.stdout.flush()
        return 1
