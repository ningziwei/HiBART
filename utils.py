import time

class Logger(object):
    def __init__(self, fp=None):
        self.fp = fp
    def __call__(self, string, end='\n'):
        curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        new_string = '[%s] ' % curr_time + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))