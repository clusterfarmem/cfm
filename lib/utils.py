import time
import subprocess
import re
import os
import argparse

g_sim_start = 0


def get_current_ts():
    global g_sim_start
    curr_ts = int(round(time.time() * 1000))

    if g_sim_start == 0:
        g_sim_start = curr_ts
        return 0

    return curr_ts - g_sim_start


def shell_exec(cmdline):
    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    return (p.returncode, out.decode('utf-8'), err.decode('utf-8'))


def check_sudo():
    if os.geteuid() != 0:
        raise RuntimeError("Run with sudo.")


def check_ratio(arg):
    ''' Check the validity of the argument passed for ratio.
        This function is passed to the argument parser.
    '''
    if arg == 'max':
        return 'max'
    else:
        try:
            value = float(arg)
        except ValueError:
            msg = "Value provided for ratio is neither a number or max"
            raise argparse.ArgumentTypeError(msg)
        if (0 < value):
            return value
        else:
            raise argparse.ArgumentTypeError("Ratio value must be > 0")


class BinTimeParser:
    def __init__(self):
        pass

    def parse(self, string):
        header = ','.join(('User Time', 'System Time',
                           'Wall Time', 'Major Page Faults'))
        values = {'User Time': self.get_user_time(string),
                  'System Time': self.get_sys_time(string),
                  'Wall Time': self.get_wall_time(string),
                  'Major Page Faults': self.get_page_faults(string)}
        return values

    def get_user_time(self, string):
        regex = re.compile(r"User time \(seconds\): (\d+.\d+)")
        return float(regex.search(string).groups()[0])

    def get_sys_time(self, string):
        regex = re.compile(r"System time \(seconds\): (\d+.\d+)")
        return float(regex.search(string).groups()[0])

    def get_wall_time(self, string):
        regex = re.compile(r"\(h:mm:ss or m:ss\): (\d*?):*(\d+):(\d+\.\d+)")
        hours, minutes, seconds = regex.search(string).groups()
        hours = float(hours) if hours else 0  # hours may be None
        minutes, seconds = float(minutes), float(seconds)
        return round(hours * 3600 + minutes * 60 + seconds, 3)

    def get_page_faults(self, string):
        regex = re.compile(r"Major \(requiring I/O\) page faults: (\d+)")
        return int(regex.search(string).groups()[0])
