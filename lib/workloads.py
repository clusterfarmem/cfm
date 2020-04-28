import threading
import subprocess
import os
import signal
import time
import psutil
import numpy as np
import shlex

from lib import utils
from lib.container import Container
from lib import constants

class Workload:
    ''' This class is not meant to be used by itself. It's only purpose
        is to provide definitions that are common to all of its children.
    '''
    # These variables are defined in child classes
    # that inherit from this class. Their definition here is
    # just done for clarity.
    wname = None
    ideal_mem = None
    min_ratio = None
    cpu_req = None

    def __init__(self, idd, pinned_cpus, mem_ratio=1):

        self.idd = idd  # a unique uint id for this workload

        # process handling
        self.thread = None
        self.popen = None
        self.stdout = None
        self.stderr = None

        # Container creation
        self.mem_ratio = mem_ratio
        self.container = Container(self.get_name(), self.ideal_mem, self.mem_ratio)
        self.container.create()
        
        # Pin CPUs
        self.pinned_cpus = pinned_cpus

        # Get shell command
        procs_path = self.container.get_procs_path()
        self.cmdline = self.get_cmdline(procs_path, pinned_cpus)

        # task timings
        self.ts_start = 0
        self.ts_finish = 0

        # Getting gradient coeffs ready
        self.percent = 0
        self.ratio = 1
        self.get_gradient()

    def __exec(self):
        " execute in self.thread "
        print(self.cmdline)

        self.ts_start = time.time()
        self.popen = subprocess.Popen(self.cmdline, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True)
        self.stdout, self.stderr = self.popen.communicate()  # blocks process exit
        assert(self.popen.returncode == 0)
        self.ts_finish = time.time()

        self.container.delete()

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()

        while not self.is_alive():
            pass

    def modify_ratio(self, new_ratio):
        self.container.set_new_size(new_ratio)

    def get_name(self):
        return self.wname + str(self.idd)

    def get_retcode(self):
        return self.popen.returncode

    def is_alive(self):
        return self.thread.is_alive() and self.popen

    def get_process_duration(self):
        return self.ts_finish - self.ts_start

    def get_usr_bin_time(self):
        ''' Parse the output of /usr/bin/time from stderr'''
        parser = utils.BinTimeParser()
        return parser.parse(self.stderr.decode('utf-8'))

    def kill(self):
        pg_id = os.getpgid(self.popen.pid)
        os.killpg(pg_id, signal.SIGKILL)
        self.thread.join()

    def set_min_ratio(self, new_min_ratio):
        self.min_ratio = new_min_ratio
        self.min_mem = self.min_ratio * self.ideal_mem

    def update(self, el_time, new_ratio, new_idd=None): # ratio = 0 is no remote memory mode
        assert el_time >= 0

        if (new_idd is not None) and self.idd == new_idd:
            assert self.percent == 0
        else:
            self.update_percent(el_time)
        self.ratio = new_ratio
        self.modify_ratio(new_ratio)

    def update_percent(self, el_time):
        self.percent = self.percent + el_time/self.profile(self.ratio)

    def profile(self,ratio): 
        return self.compute_ratio_from_coeff(self.coeff, ratio)*1000 # from second to millisecond

    def get_gradient(self):
        tmp_coeff = self.coeff + [0]
        self.gd_coeff = np.polyder(self.coeff)
        self.mem_gd_coeff = np.polyder(tmp_coeff)

    def gradient(self, ratio):
        return self.compute_ratio_from_coeff(self.gd_coeff, ratio)

    def mem_gradient(self,ratio):
        return self.compute_ratio_from_coeff(self.mem_gd_coeff, ratio)

    def compute_ratio_from_coeff(self, coeffs, ratio):
        p = 0
        order = len(coeffs)
        for i in range(order):
            p += coeffs[i] * ratio**(order-1-i)
        return p

    def get_pids(self):
        return self.container.get_pids()

class Quicksort(Workload):
    wname = "quicksort"
    ideal_mem = 8250
    min_ratio = 0.65
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "quicksort"
    cpu_req = 1
    coeff = [-1984.129, 4548.033, -3588.554, 1048.644, 252.997]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} &&".format(procs_path)
        arg = '8192'
        shell_cmd = '/usr/bin/time -v' + ' ' + constants.WORK_DIR + '/quicksort/quicksort {}'.format(arg)
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))
        return full_command


class Linpack(Workload):
    wname = "linpack"
    ideal_mem = 1600
    min_ratio = 0.9
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "xlinpack_xeon64"
    cpu_req = 4
    coeff = [38.52, -77.88, 26.86, 36.70]

    def get_cmdline(self, procs_path, pinned_cpus):
        linpack_dir = constants.WORK_DIR + '/linpack'
        prefix = "echo $$ > {} &&".format(procs_path)
        set_vars = ' '.join(('MKL_NUM_THREADS=4',
                             'OMP_NUM_THREADS=4',
                             'MKL_DOMAIN_NUM_THREADS=4'))
        
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)

        set_vars = ' '.join(('KMP_AFFINITY=nowarnings,compact,1,0,granularity=fine',
                             set_vars))

        bin_path = '{}/xlinpack_xeon64'.format(linpack_dir)
        cmdline = '{}/lininput_xeon64'.format(linpack_dir)
        after_exec = ' '.join(('/usr/bin/time -v', bin_path, cmdline))
        full_command = ' '.join((prefix, set_vars, 'exec', set_cpu, after_exec))
        return full_command


class Tfinception(Workload):
    wname = "tf-inception"
    ideal_mem = 2120
    min_ratio = 0.9
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "python3"
    cpu_req = 2
    coeff = [-1617.416, 3789.953, -2993.734, 1225.477]

    def get_cmdline(self, procs_path, pinned_cpus):
        work_dir = ''.join((constants.WORK_DIR,
                            '/tensorflow/benchmarks/scripts/tf_cnn_benchmarks'))
        
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)

        cd_dir = ' '.join(('cd', work_dir, '&&'))
        prefix = "echo $$ > {} &&".format(procs_path)
        set_vars = ' '.join(('KMP_BLOCK_TIME=0',
                             'KMP_SETTINGS=1 OMP_NUM_THREADS=2'))
        
        set_vars = ' '.join(('KMP_AFFINITY=granularity=fine,verbose,compact,1,0',
                             set_vars))

        shell_cmd = ' '.join(("/usr/bin/time -v python3 tf_cnn_benchmarks.py",
                              "--forward_only=True --data_format=NHWC --device=cpu",
                              "--batch_size=64 --num_inter_threads=1",
                              "--num_intra_threads=2 --nodistortions",
                              "--model=inception3",
                              "--kmp_blocktime=0 --num_batches=20",
                              "--num_warmup_batches 0"))
        full_command = ' '.join((cd_dir, prefix, set_vars, 'exec', set_cpu, shell_cmd))
        return full_command


class Tfresnet(Workload):
    wname = "tf-resnet"
    ideal_mem = 1268
    min_ratio = 0.9
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "python3"
    cpu_req = 2
    coeff = [-1617.416, 3789.953, -2993.734, 1225.477]

    def get_cmdline(self, procs_path, pinned_cpus):
        work_dir = ''.join((constants.WORK_DIR,
                            '/tensorflow/benchmarks/scripts/tf_cnn_benchmarks'))

        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        
        cd_dir = ' '.join(('cd', work_dir, '&&'))
        prefix = "echo $$ > {} &&".format(procs_path)
        set_vars = ' '.join(('KMP_BLOCK_TIME=0',
                             'KMP_SETTINGS=1 OMP_NUM_THREADS=2'))
        
        set_vars = ' '.join(('KMP_AFFINITY=granularity=fine,verbose,compact,1,0',
                             set_vars))

        shell_cmd = ' '.join(("/usr/bin/time -v python3 tf_cnn_benchmarks.py",
                              "--forward_only=True --data_format=NHWC --device=cpu",
                              "--batch_size=64 --num_inter_threads=1",
                              "--num_intra_threads=2 --nodistortions",
                              "--model=resnet50",
                              "--kmp_blocktime=0 --num_batches=20",
                              "--num_warmup_batches 0"))
        full_command = ' '.join((cd_dir, prefix, set_vars, 'exec', set_cpu, shell_cmd))
        return full_command


class Kmeans(Workload):
    wname = "kmeans"
    ideal_mem = 4847
    binary_name = "python3"
    min_ratio = 0.75
    min_mem = int(min_ratio * ideal_mem)
    cpu_req = 1
    coeff = [-10341.875,  31554.403, -34346.894,  15214.428,  -1730.533]

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = "echo $$ > {} && OMP_NUM_THREADS={}".format(procs_path, self.cpu_req)
        bin_path = constants.WORK_DIR + '/kmeans/kmeans.py'
        shell_cmd = '/usr/bin/time -v python3' + ' ' + bin_path
        
        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)
        
        full_command = ' '.join((prefix, 'exec', set_cpu, shell_cmd))

        return full_command


class Spark(Workload):
    wname = "spark"
    ideal_mem = 4400
    min_ratio = 0.75
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "java"
    cpu_req = 3
    coeff = [4689.05, -10841.59, 7709.92, -1486.13]

    def get_cmdline(self, procs_path, pinned_cpus):
        target_dir = ''.join((constants.WORK_DIR, '/spark/pagerank'))
        cd_dir = ' '.join(('cd', target_dir, '&&'))
        prefix = 'echo $$ > {} &&'.format(procs_path)

        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)

        shell_cmd = ' '.join(('/usr/bin/time -v',
                              constants.SPARK_HOME + 'bin/spark-submit',
                              '--driver-memory 10g',
                              '--class \"pagerank\"',
                              '--master local[2]',
                              'target/scala-2.11/pagerank_2.11-1.0.jar'))
        full_command = ' '.join((cd_dir, prefix, 'exec', set_cpu, shell_cmd))
        return full_command

class Memaslap(Workload):
    wname = "memaslap"
    ideal_mem = 12288
    min_ratio = 0.5
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "memcached"
    port_number = 11211
    cpu_req = 2
    coeff = [-11626.894, 32733.914, -31797.375, 11484.578, 113.33]

    def __init__(self, idd, pinned_cpus, mem_ratio=1):
        super().__init__(idd, pinned_cpus, mem_ratio)
        self.port_number = Memaslap.port_number
        self.memaslap_pids = set()
        Memaslap.port_number += 1

    def get_cmdline(self, procs_path, pinned_cpus):
        prefix = 'echo $$ > {} &&'
        memcached_serv = "/usr/bin/time -v memcached -l localhost -p {} -m {} -t 1".format(self.port_number, 
                                                        self.ideal_mem)
        cpu_list = list(pinned_cpus)
        taskset_serv = 'taskset -c {}'.format(cpu_list[0])
        memcached_serv = ' '.join((prefix, 'exec', taskset_serv, memcached_serv))
        memcached_serv = memcached_serv.format(procs_path)

        taskset_memaslap = 'taskset -c {}'.format(cpu_list[1])
        memaslap_fill = taskset_memaslap + ' ' + "memaslap -s localhost:{} -T 1 -F {} --execute_number 30000000"
        memaslap_fill = memaslap_fill.format(self.port_number, "memaslap/memaslap_fill")

        memaslap_query = taskset_memaslap + ' ' + "memaslap -s localhost:{} -T 1 -F {} --execute_number 100000000"
        memaslap_query = memaslap_query.format(self.port_number, "memaslap/memaslap_etc")
        sleep = 'sleep 5'
        memaslap_cmd = ' && '.join((memaslap_fill, sleep, memaslap_query))
        return (memcached_serv, memaslap_fill, memaslap_query)

    def start(self):
        self.thread = threading.Thread(target=self.__exec)
        self.thread.start()

        while not self.is_alive():
            pass

    def __exec(self):
        memcached, memaslap_fill, memaslap_query = self.cmdline

        " execute in self.thread "
        print(self.cmdline)

        self.ts_start = time.time()

        self.popen = subprocess.Popen(memcached, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, shell=True,
                                      preexec_fn=os.setsid)

        time.sleep(3) # Wait for memcached to boot
        memaslap_proc = subprocess.Popen(shlex.split(memaslap_fill), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, shell=False)
        self.memaslap_pids.add(memaslap_proc.pid)
        stdout, stderr = memaslap_proc.communicate()
        self.memaslap_pids.remove(memaslap_proc.pid)

        time.sleep(5)
        memaslap_proc = subprocess.Popen(shlex.split(memaslap_query), stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, shell=False)
        self.memaslap_pids.add(memaslap_proc.pid)
        stdout, stderr = memaslap_proc.communicate()
        self.memaslap_pids.remove(memaslap_proc.pid)
        
        print(stdout.decode('utf-8'))
        print(stderr.decode('utf-8'))

        os.killpg(os.getpgid(self.popen.pid), signal.SIGINT)

        self.stdout, self.stderr = self.popen.communicate()
        self.ts_finish = time.time()
        print(self.stdout.decode('utf-8'))
        print(self.stderr.decode('utf-8'))

        self.container.delete()

    def get_pids(self):
        pids = list(self.container.get_pids())
        pids.extend(self.memaslap_pids)
        return pids

class Stream(Workload):
    wname = "stream"
    ideal_mem = 4150
    min_ratio = 0.50
    min_mem = int(min_ratio * ideal_mem)
    binary_name = "stream_c.exe"
    cpu_req = 1
    coeff = [0]

    def get_cmdline(self, procs_path, pinned_cpus):
        target_dir = ''.join((constants.WORK_DIR, '/stream'))
        cd_dir = ' '.join(('cd', target_dir, '&&'))
        prefix = 'echo $$ > {} && OMP_NUM_THREADS={}'.format(procs_path, len(pinned_cpus))

        pinned_cpus_string = ','.join(map(str, pinned_cpus))
        set_cpu = 'taskset -c {}'.format(pinned_cpus_string)

        shell_cmd = 'nice -n -2 /usr/bin/time -v ./stream_c.exe'.format(len(pinned_cpus))
        full_command = ' '.join((cd_dir, prefix, 'exec', set_cpu, shell_cmd))
        return full_command

def get_workload_class(wname):
    return {'quicksort': Quicksort,
            'linpack': Linpack,
            'tf-inception': Tfinception,
            'tf-resnet': Tfresnet,
            'spark': Spark,
            'kmeans': Kmeans,
            'memaslap': Memaslap,
            'stream': Stream}[wname]
