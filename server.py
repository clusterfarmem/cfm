#!/usr/bin/python3
"""Server receives connection from scheduler"""

from concurrent import futures
import time
import logging
import argparse
import socket
from tensorflow.python.framework import test_util

import multiprocessing
import psutil

import grpc

from protocol import protocol_pb2
from protocol import protocol_pb2_grpc

from lib import workloads

import re

import numpy as np
from scipy.optimize import Bounds, minimize

MAIN_LOOP_SLEEP = 1
DRIVER_PATH = "/sys/class/infiniband/mlx4_0/ports/1/counters/{}"
MEGABYTE = 1024*1024
CURR_PAGES_PATH = '/sys/kernel/debug/frontswap/curr_pages'
SWAPPINESS_PATH = '/proc/sys/vm/swappiness'
THP_PATH = "/sys/kernel/mm/transparent_hugepage/enabled"
SOMAXCONN_PATH = "/proc/sys/net/core/somaxconn"
SWAPPINESS_THRESHOLD = 60
SWAP_REGEX = re.compile(rb"VmSwap:\s+(\d+)\s+\.*")


def eq(x,mems,local_mem):
    return np.dot(x, mems) - local_mem

def eq_grad(x,mems,local_mem):
    return mems

def obj_new(x, ideal_mems, percents, profiles, gradients=None, mem_gradients=None, beta=0):
    r1 = 0
    r2 = 0 
    r3 = 0
    r4 = 0 
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
        r3 += ideal_mems[i]*(1-percents[i])*x[i]*profiles[i](x[i])/1000
        r4 += ideal_mems[i]*(1-percents[i])*profiles[i](1)/1000
    return r1/r2 + beta*r3/r4 

def obj_grad_new(x, ideal_mems, percents, profiles, gradients, mem_gradients, beta=0):
    r1 = 0
    r2 = 0 
    r4 = 0
    g1 = np.empty(ideal_mems.shape)
    g2 = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
        r4 += ideal_mems[i]*(1-percents[i])*profiles[i](1)/1000

        g1[i] = ideal_mems[i]*(1-percents[i])*mem_gradients[i](x[i]) 
        g2[i] = ideal_mems[i]*(1-percents[i])*(gradients[i](x[i]) - mem_gradients[i](x[i]))
    
    grads = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]): 
        grads[i] = (g1[i]*r2 - r1*g2[i])/r2**2 + beta*g1[i]/r4 # r3 has the same gradient as r1
    return grads 

class Machine:
    def __init__(self):
        self.total_cpus = 0 # number of cpus this machine can use
        self.free_cpus = 0
        self.total_mem = 0 # amount of memory this machine can use
        self.alloc_mem = 0
        self.min_mem_sum = 0
        self.cur_ratio = 1

        # how much memory we have placed in this machine.
        # can be > total_mem when using remote memory
        self.remote_mem = False
        self.executing = []
        self.finished = []
        self.running = False
        self.shutdown_now = False
        self.using_remote_mem = False

        # Sampling
        self.cpu_samples = []
        self.mem_samples = []
        self.swap_samples = []
        self.bw_in_samples = []
        self.bw_out_samples = []
        self.bytes_in_samples = 0
        self.bytes_out_samples = 0
        self.curr_pages = []

        # Bandwidth state
        self.prev_recv = 0
        self.prev_sent = 0

        # State for calculating percents
        self.last_time = 0
        self.slow_downs = {}
        for wname in ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']:
            self.slow_downs[wname] = 1

    def checkin(self, max_cpus, max_mem, use_remote, uniform_ratio, variable_ratios, limit_remote_mem, optimal):
        """
        the scheduler checks in with these params.
        we return whether we have enough resources to do the checkin.
        if True, this machine will start executing jobs
        """
        machine_cpus = multiprocessing.cpu_count()
        machine_mem = psutil.virtual_memory().total / 1024 / 1024

        if max_cpus > machine_cpus or max_mem > machine_mem:
            logging.info("Checkin Unsuccessful")
            return False

        # the checkin used feasible num. of cpus and mem. now initialize
        # the machine resources
        self.total_mem = max_mem
        self.total_cpus = max_cpus
        self.free_cpus = max_cpus
        self.remote_mem = use_remote
        self.uniform_ratio = uniform_ratio
        self.running = True
        self.variable_ratios = variable_ratios
        self.limit_remote_mem = limit_remote_mem
        self.unpinned_cpus = set(range(self.total_cpus))
        self.cpu_assignments = {c: None for c in self.unpinned_cpus}
        self.base_time = time.time()
        self.reclaimer_cpu = self.total_cpus - 1

        self.optimal = optimal

        if self.remote_mem:
            try:
                with open(DRIVER_PATH.format("port_xmit_data")) as tx_file:
                    tx_bytes = int(tx_file.read()) * 4
            except FileNotFoundError:
                    tx_bytes = 0
            
            try:       
                with open(DRIVER_PATH.format("port_rcv_data")) as recv_file:
                    recv_bytes = int(recv_file.read()) * 4
            except FileNotFoundError:
                    recv_bytes = 0
                    
            self.prev_sent = tx_bytes
            self.prev_recv = recv_bytes

            logging.info("Initial tx value: {}".format(tx_bytes / MEGABYTE))
            logging.info("Initial recv value: {}".format(tx_bytes / MEGABYTE))


        #self.check_swappiness()
        self.check_thp()
        self.check_somaxconn()
        self.check_tf_mkl()

        logging.info("Checkin Successful")

        return True

    def check_state(self):
        if self.using_remote_mem:
            if self.alloc_mem <= self.total_mem:
                self.using_remote_mem = False
                print("Transitioning to 8 cpus")
        else:
            if self.alloc_mem > self.total_mem:
                self.using_remote_mem = True
                print("Transitioning to 7 cpus")

    def check_reclaimer_cpu(self): # Check if reclaimer CPU is being used and move workload off of it
        all_cpus = set(range(self.reclaimer_cpu)) # All CPUs except the reclaimer
        pinnable_cpus = self.unpinned_cpus.intersection(all_cpus) # Only the CPUs that aren't executing
        
        ''' We're now using far memory but a workload is executing on
            the reclaimer CPU. Need to move it off.'''
        if self.cpu_assignments[self.reclaimer_cpu]:
            workload_on_reclaimer = self.cpu_assignments[self.reclaimer_cpu]
            pids = workload_on_reclaimer.get_pids() # Potentially offending pids
            replacement_cpu = pinnable_cpus.pop() # Get a replacement CPU
            print("Moving {} off of the reclaimer CPU".format(workload_on_reclaimer.get_name()))
            
            ''' Not just the parent. But the children too'''
            for pid in pids:
                process = psutil.Process(pid)
                affinity_list = process.cpu_affinity() # 
                if self.reclaimer_cpu in affinity_list:
                    print("Moving {} off of the reclaimer CPU and to {}".format(pid, replacement_cpu))
                    new_affinity_list = [cpu for cpu in affinity_list if cpu != self.reclaimer_cpu]
                    new_affinity_list.append(replacement_cpu)
                    process.cpu_affinity(new_affinity_list)
            
            self.cpu_assignments[self.reclaimer_cpu] = None
            self.cpu_assignments[replacement_cpu] = workload_on_reclaimer
            self.unpinned_cpus.remove(replacement_cpu)
            self.unpinned_cpus.add(self.reclaimer_cpu)
            existing_pinned_cpus = set(workload_on_reclaimer.pinned_cpus)
            existing_pinned_cpus.remove(self.reclaimer_cpu)
            existing_pinned_cpus.add(replacement_cpu)
            workload_on_reclaimer.pinned_cpus = existing_pinned_cpus
        
        return pinnable_cpus
        
    def wait_for_swap_to_fall(self):
        start = time.time()
        while True:
            allowed_far = max(0, self.alloc_mem - self.total_mem)
            allowed_far = 1024 if allowed_far == 0 else allowed_far
            far_mem = self.get_swap()
            print("allowed_far={} far_mem={}".format(allowed_far, far_mem))

            if far_mem <= allowed_far or far_mem < 32:
                break

            if time.time() - start > 20:
                print("waited for 20 seconds. let it go")
                break

            print("wait for swap usage to go down")
            time.sleep(0.5)
        end = time.time()
        print('waited for {} s'.format(end - start))
        global total_wait_time
        total_wait_time += end - start      

    def execute(self, new_workload_name, idd):
        new_workload_class = workloads.get_workload_class(new_workload_name)
        self.alloc_mem += new_workload_class.ideal_mem
        self.check_state() # Update self.using_remote_mem
        
        if self.using_remote_mem:
            pinnable_cpus = self.check_reclaimer_cpu()
        else:
            pinnable_cpus = set(self.unpinned_cpus)
        
        new_workload_cpus = set([pinnable_cpus.pop() for i in range(new_workload_class.cpu_req)])
        self.unpinned_cpus.difference_update(new_workload_cpus) # Remove these cpus from the unpinned set
        new_workload = new_workload_class(idd, new_workload_cpus)

        for cpu in new_workload_cpus:
            self.cpu_assignments[cpu] = new_workload

        if new_workload_name in self.variable_ratios:
            new_workload.set_min_ratio(self.variable_ratios[new_workload_name])

        self.min_mem_sum += new_workload.min_mem
        self.free_cpus -= new_workload_class.cpu_req


        all_workloads = self.executing + [new_workload]

        if self.remote_mem:
            if self.uniform_ratio:
                self.shrink_all_uniformly(all_workloads)
            elif self.optimal:
                self.shrink_all_optimally(all_workloads, idd)
                self.last_time = time.time() * 1000 # to ms
            else:
                self.shrink_all_proportionally(all_workloads)

        else:
            assert self.alloc_mem <= self.total_mem

        assert self.free_cpus >= 0

        new_workload.start()
        self.executing.append(new_workload)
        print("started {} at {} s".format(new_workload.get_name(), round(new_workload.ts_start - self.base_time, 3)))

    def check_swappiness(self):
        with open(SWAPPINESS_PATH, 'r') as f:
            swappiness = int(f.read())

        assert(not self.remote_mem or swappiness >= SWAPPINESS_THRESHOLD),\
            "Swappiness needs to be >= {} when using remote mem".format(SWAPPINESS_THRESHOLD)
        
        assert(self.remote_mem or swappiness == 1),\
            "Swappiness needs to be == 1 when not using remote mem"
    
    def check_thp(self):
        with open(THP_PATH, 'r') as f:
            assert('[never]' in f.read()), 'Transparent Hugepage is not disabled' 

    def check_somaxconn(self):
        with open(SOMAXCONN_PATH, 'r') as f:
            assert('65536' == f.read().strip('\n')), 'somaxconn is set to an incorrect value'
    
    def check_tf_mkl(self):
        assert(test_util.IsMklEnabled()), "tensorflow doesn't have mkl enabled"

    def set_cur_ratio(self):
        try:
            # Ratio > 1 means that we're haven't fully utilized local memory
            self.cur_ratio = min(1, self.total_mem / self.alloc_mem)
        except ZeroDivisionError:
            self.cur_ratio = 1

    def shrink_all_uniformly(self, workloads):
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        try:
            local_ratio = min(1, self.total_mem / total_ideal_mem)
        except ZeroDivisionError:
            local_ratio = 1

        assert local_ratio >= self.uniform_ratio
        self.set_cur_ratio()

        for w in workloads:
            w.modify_ratio(local_ratio)

    def shrink_all_proportionally(self, workloads):
        assert self.min_mem_sum <= self.total_mem

        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        total_min_mem = sum([w.min_mem for w in workloads])

        memory_pool = total_ideal_mem - total_min_mem

        # Prevent containers from overgrowing
        excess_mem = max(0, total_ideal_mem - self.total_mem)

        # Shrink each container
        for w in workloads:
            try:
                share_of_excess = (w.ideal_mem - w.min_mem) / memory_pool * excess_mem
            except ZeroDivisionError:
                # The pool of memory allowed to be pushed to remote storage is empty
                share_of_excess = 0
            ratio = (w.ideal_mem - share_of_excess) / w.ideal_mem
            w.modify_ratio(ratio)

    def shrink_all_optimally(self, workloads, new_idd=None):
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        total_min_mem = sum([w.min_mem for w in workloads])
        memory_pool = total_ideal_mem - total_min_mem

        excess_mem = max(0, total_ideal_mem - self.total_mem)

        # Shrink each container
        init_ratios = []
        for w in workloads:
            try:
                share_of_excess = (w.ideal_mem - w.min_mem) / memory_pool * excess_mem
            except ZeroDivisionError:
                # The pool of memory allowed to be pushed to remote storage is empty
                share_of_excess = 0
            ratio = (w.ideal_mem - share_of_excess) / w.ideal_mem
            init_ratios.append(ratio)

        if excess_mem <= 0:
            opt_ratios = init_ratios
        else:
            ratios,_ = self.compute_opt_ratios(workloads,init_ratios, new_idd)
            opt_ratios = ratios.tolist()
        
        if self.last_time == 0:
            el_time = 0
        else:
            el_time = time.time()*1000 - self.last_time

        for w,ratio in zip(workloads,opt_ratios):
            w.update(el_time, ratio, new_idd)

    def compute_opt_ratios(self, workloads, init_ratios, new_idd):
        el_time = time.time()*1000 - self.last_time
        ideal_mems = np.array([w.ideal_mem for w in workloads])
        percents = np.array([(1-(w.idd==new_idd))*min( (w.percent+el_time/w.profile(w.ratio))/self.slow_downs[w.wname], 0.95) for w in workloads])
        profiles = [w.profile for w in workloads]
        mem_gradients = [w.mem_gradient for w in workloads]
        gradients = [w.gradient for w in workloads]

        x0 = np.array(init_ratios)

        eq_cons = {'type': 'eq',  'fun' : eq, 'jac': eq_grad, 'args': (ideal_mems,self.total_mem)}
        bounds = Bounds(0.5, 1.0)
        beta = 0
        res = minimize(obj_new, x0, method='SLSQP', jac=obj_grad_new, args=(ideal_mems, percents, profiles, gradients, mem_gradients, beta), constraints=eq_cons, options={'disp': False}, bounds=bounds)
        final_ratios = res.x
        return np.round(final_ratios,3), res.fun

    def check_finished(self):
        new_finished = []
        old_alloc_mem = self.alloc_mem
        for workload in self.executing[:]:
            if not workload.is_alive():
                finished_string = "{} finished at {} s (duration={})"
                print(finished_string.format(workload.get_name(),
                                            round(workload.ts_finish - self.base_time, 3),
                                            workload.get_process_duration()))
                
                self.unpinned_cpus.update(workload.pinned_cpus)
                
                for cpu in workload.pinned_cpus:
                    self.cpu_assignments[cpu] = None
                self.free_cpus += workload.cpu_req
                self.alloc_mem -= workload.ideal_mem
                self.min_mem_sum -= workload.min_mem
                self.executing.remove(workload)
                new_finished.append(workload)
                
                # adjust percents
                el_time = time.time()*1000 - self.last_time
                final_percent = workload.percent + el_time/workload.profile(workload.ratio)
                if workload.wname in self.slow_downs:
                    self.slow_downs[workload.wname] = 0.05*final_percent + 0.95*self.slow_downs[workload.wname]
                    logging.info('{} new slow down is {}'.format(workload.wname,self.slow_downs[workload.wname]))
        self.finished.extend(new_finished)

        if new_finished:
            print("{} tasks finished".format(len(new_finished)))
            if self.remote_mem:
                if self.uniform_ratio:
                    self.shrink_all_uniformly(self.executing)
                elif self.optimal:
                    self.shrink_all_optimally(self.executing, None)
                    self.last_time = time.time()*1000
                else:
                    self.shrink_all_proportionally(self.executing)

        self.check_state()

    def clear_finished(self):
        self.finished = []

    def get_resources(self):
        return {'free_cpus': self.free_cpus,
                'alloc_mem': self.alloc_mem,
                'min_mem_sum': self.min_mem_sum}

    def shutdown(self):
        for workload in self.executing:
            print("Terminating {}".format(workload.get_name()))
            workload.kill()
        self.shutdown_now = True
        print("Shutting Down")

    def get_swap(self):
        # Get list of pids
        pids = list()
        for workload in self.executing:
            '''Only get pids for things in the container
               This prevents the memaslap from being included with memcached'''
            pids.extend(workload.container.get_pids())

        total_swap = 0
        for pid in pids:
            try:
                path = '/proc/{}/status'.format(pid)
                with open(path, 'rb', buffering=0) as f:
                    swap = int(SWAP_REGEX.findall(f.read())[0])
                total_swap += swap
            except Exception:
                continue
        total_swap = total_swap / 1024 # Convert from KB to MB
        return total_swap

    def sample(self):
        if self.running:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            swap = self.get_swap()

            # get bandwidth measurements
            if self.remote_mem:
                try:
                    with open(DRIVER_PATH.format("port_xmit_data")) as tx_file:
                        tx_bytes = int(tx_file.read()) * 4
                except FileNotFoundError:
                        tx_bytes = 0
                try:
                    with open(DRIVER_PATH.format("port_rcv_data")) as recv_file:
                        recv_bytes = int(recv_file.read()) * 4
                except FileNotFoundError:
                        recv_bytes = 0

                bw_tx = tx_bytes - self.prev_sent
                bw_recv = recv_bytes - self.prev_recv

                try:
                    with open(CURR_PAGES_PATH, 'r') as f_curr_pages:
                        curr_pages = int(f_curr_pages.read())
                except FileNotFoundError:
                        curr_pages = 0

            stats = "CPU: {}, Total Mem: {}, Used Mem: {}, Used Swap: {}".format(cpu,
                    mem.total, mem.used, round(swap, 3))

            logging.info(stats)

            self.cpu_samples.append(cpu)
            self.mem_samples.append(mem.used / mem.total * 100)
            self.swap_samples.append(swap)
            if self.remote_mem:
                self.bw_in_samples.append(bw_recv)
                self.bw_out_samples.append(bw_tx)
                self.bytes_in_samples += bw_recv
                self.bytes_out_samples += bw_tx
                self.prev_recv = recv_bytes
                self.prev_sent = tx_bytes
                self.curr_pages.append(curr_pages)

                logging.info("bw_tx: {}".format(bw_tx / MEGABYTE))
                logging.info("bw_recv: {}".format(bw_recv / MEGABYTE))
        else:
            pass


class Scheduler(protocol_pb2_grpc.SchedulerServicer):
    def __init__(self, machine, servername):
        self.machine = machine
        self.name = servername

    def checkin(self, req, context):
        success = self.machine.checkin(req.max_cpus, req.max_mem,
                                       req.use_remote_mem, req.uniform_ratio,
                                       req.variable_ratios, req.limit_remote_mem, req.optimal)

        return protocol_pb2.CheckinReply(server_name=self.name, success=success)

    def execute(self, request, context):
        """ executes the request.wname workload.
        if we are using remote memory, computes the new ratio
        that will be required after placing the workload."""
        self.machine.check_finished()
        self.machine.execute(request.wname, request.idd)
        return protocol_pb2.ExecuteReply(success=True)

    def get_resources(self, request, context):
        self.machine.check_finished()
        resources = self.machine.get_resources()
        # ** Expands dictionary into named arguments for a function
        reply = protocol_pb2.GetResourcesReply(**resources)
        return reply

    def get_finished(self, request, context):
        self.machine.check_finished()
        start_times = {f.idd: f.ts_start - self.machine.base_time
                          for f in self.machine.finished}
        finished_times = {f.idd: f.ts_finish - self.machine.base_time
                          for f in self.machine.finished}
        reply = protocol_pb2.GetFinishedReply(start_times = start_times,
                                              finished_times=finished_times)
        self.machine.clear_finished()
        return reply

    def shutdown(self, request, context):
        self.machine.shutdown()
        reply = protocol_pb2.ShutdownReply(success=True)
        return reply

    def get_samples(self, request, context):
        reply = protocol_pb2.GetSamplesReply()
        reply.cpu_util.extend(self.machine.cpu_samples)
        reply.mem_util.extend(self.machine.mem_samples)
        reply.swap_util.extend(self.machine.swap_samples)
        reply.curr_pages.extend(self.machine.curr_pages)

        bw_in_mb = map(lambda x: x / MEGABYTE, self.machine.bw_in_samples)
        reply.bw_in.extend(bw_in_mb)
        bw_out_mb = map(lambda x: x / MEGABYTE, self.machine.bw_out_samples)
        reply.bw_out.extend(bw_out_mb)

        reply.bytes_in = self.machine.bytes_in_samples / MEGABYTE
        reply.bytes_out = self.machine.bytes_out_samples / MEGABYTE
        return reply

def serve():
    hostname = socket.gethostname()
    thismachine = Machine()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scheduler = Scheduler(thismachine, hostname)
    protocol_pb2_grpc.add_SchedulerServicer_to_server(scheduler, server)

    server.add_insecure_port('[::]:50051')
    server.start()

    total_cpus = multiprocessing.cpu_count()
    total_mem = psutil.virtual_memory().total
    print("server {} waiting for connection, avail cpus={} mem={} MB".format(hostname,
                                              total_cpus, int(total_mem/(1024*1024))))

    try:
        while not thismachine.shutdown_now:
            t0 = time.time()
            thismachine.sample()
            t1 = time.time()
            time.sleep(max(0, MAIN_LOOP_SLEEP - (t1 - t0)))
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true',
                        help='Write out log to file')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', filename='log.txt', level=logging.DEBUG, filemode='w')
    else:
        logging.basicConfig()
    
    total_wait_time = 0
    serve()
    print('total wait tims: {} s'.format(total_wait_time))
