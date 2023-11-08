import os
from os import path
import logging
import datetime
import sys

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    def __init__(self, args, save_dir):
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        self.args = args

        cur_time = get_local_time()
        args.exp_time = cur_time

        if not path.exists(save_dir):
            os.makedirs(save_dir)

        if args.log:
            log_file = logging.FileHandler(path.join(save_dir, 'train_log_{}.txt'.format(cur_time)))
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            log_file.setFormatter(formatter)
            self.logger.addHandler(log_file)
            
        self.log(f"PID: {os.getpid()}")

        # log command that runs the code
        s = ""
        for arg in sys.argv:
            s = s + arg + " "
        self.log(os.path.basename(sys.executable) + " " + s)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)