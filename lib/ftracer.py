import sys
import re
from multiprocessing import cpu_count
from lib import constants
from lib import utils

BUFFER_SIZE_DEFAULT = 1408
BUFFER_SIZE_MAX = 500000


class FTracer:
    def __init__(self, filter_functions):
        self.filter_functions = filter_functions

    def read_trace_stats(self):
        regex = re.compile(r'([^\s]+)\s+(\d+)\s+(\d+\.*\d*)'
                           '\s+us\s+(\d+\.*\d*)\s+us\s+(\d+\.*\d*)')
        stats = dict()

        for cpu in range(cpu_count()):
            filename = constants.TRACING_DIR + 'trace_stat/function' + str(cpu)
            with open(filename, 'r') as f:
                for line in f:
                    match = regex.search(line)
                    if match:
                        func_name, hit, time, avg, std_dev = match.groups()
                        if func_name in stats:
                            stats[func_name]['hits'] += int(hit)
                            stats[func_name]['sum_time'] += float(time)
                        else:
                            stats[func_name] = {'hits': int(hit),
                                                'sum_time': float(time)}
                    else:
                        pass
        for func, values in stats.items():
            values['avg'] = values['sum_time']/values['hits']
        return stats

    def set_ftrace_filter(self):
        filter_file = constants.TRACING_DIR + 'set_ftrace_filter'
        with open(filter_file, 'w') as f:
            f.write('\n'.join(self.filter_functions))

    def enable_function_profile(self):
        filename = constants.TRACING_DIR + 'function_profile_enabled'
        with open(filename, 'w') as f:
            f.write('1')

    def disable_function_profile(self):
        filename = constants.TRACING_DIR + 'function_profile_enabled'
        with open(filename, 'w') as f:
            f.write('0')

    def set_buffer_size_kb(self, size):
        with open(constants.TRACING_DIR + 'buffer_size_kb', 'w') as f:
            f.write(str(size))

    def enable_tracing_on(self):
        with open(constants.TRACING_DIR + 'tracing_on', 'w') as f:
            f.write('1')

    def disable_tracing_on(self):
        with open(constants.TRACING_DIR + 'tracing_on', 'w') as f:
            f.write('0')

    def set_current_tracer(self, tracer):
        with open(constants.TRACING_DIR + 'current_tracer', 'w') as f:
            f.write(tracer)

    def copy_trace(self, name, mem_ratio):
        print("Copying trace to current directory")
        cp_trace = ' '.join(('sudo cp',
                             constants.TRACING_DIR + 'trace',
                             '{}_{}_{}')).format(name, mem_ratio,
                                                 '_'.join(self.filter_functions))
        utils.shell_exec(cp_trace)

    def setup_profile(self):
        self.set_current_tracer('function')
        self.set_ftrace_filter()
        self.disable_function_profile()
        self.enable_function_profile()

    def teardown_profile(self):
        self.disable_function_profile()

    def setup_timestamp(self):
        self.set_current_tracer('function')
        self.set_ftrace_filter()
        self.set_buffer_size_kb(BUFFER_SIZE_MAX)
        self.enable_tracing_on()

    def teardown_timestamp(self):
        self.disable_tracing_on()
        self.set_buffer_size_kb(BUFFER_SIZE_DEFAULT)
