# Setup and pre-requisites

On the client node the fastswap kernel and driver must be loaded. On the far memory node the server binary `rmserver` must be running. Please see https://github.com/clusterfarmem/fastswap for more details.

## General pre-requisites

You'll need python3, grpcio, grpcio-tools, numpy and scipy to execute various parts of our framework. Please make sure your python environment can see these modules.

## Workload setup (for single and multi-workload benchmarks)

* quicksort
    * Change directory to quicksort and type make
* linpack
    * No setup required, but most likely you'll need an Intel CPU
* tf-inception
    * tensorflow 1.14 is required
    * Init submodules `git submodule update --init`
* spark
    * We assume the user has installed [spark 2.4](https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz) at `~/spark-2.4.0-bin-hadoop2.7`
* kmeans
    * Requires sklearn available in python3
* memcached
    * Requires `memcached` and `memaslap` to be installed and available in your $PATH environment.
* stream
    * Change directory to stream and type make

## Setting up cgroups
### Disable cgroup v1
* Open /boot/grub/grub.cfg in your editor of choice
* Find the `menuentry` for the fastswap kernel
* Add `cgroup_no_v1=memory` to the end of the line beginning in `linux   /boot/vmlinuz-4.11.0-sswap`
* Save and exit the file
* Run: sudo update-grub
* Reboot

### Enable cgroup v2
The framework and scripts rely on the cgroup system to be mounted at /cgroup2. Perform the following actions:
* Run `sudo mkdir /cgroup2` to create root mount point
* Execute `setup/init_bench_cgroups.sh`
    * Mounts cgroup system
    * Changes ownership of the mount point (and all nested files) to the current user
    * Enables prefetching

## Protocol Buffers
We use [the grpc framework](https://grpc.io) and [protocol buffers](https://developers.google.com/protocol-buffers/docs/pythontutorial) to communicate between the scheduler and servers. The messages that we've defined are in `protocol/protocol.proto`. To generate them the corresponding `.py` files, execute the following command in the `protocol` directory:
    
    source gen_protocol.sh

# Single Workload Benchmarks
## `benchmark.py`

`Benchmark.py` is the command center from which you can run local, single benchmarks. It accepts numerous arguments but only two, `workload` and `ratio`, are required. Its minimum invocation is the following:

    ./benchmark.py <workload> <ratio>

Where `workload` is an application that the toolset has been configured to benchmark (Ex: linpack) and `ratio` is the portion of its resident set size that you want to keep in local memory, expressed as a decimal.

Running the tool in this way will set the appropriate limits in the applications cgroup, run it to completion, then print statistics to stdout.

## Arguments
Argument            | Description | Required
--------------------------------|-----------------------|----------------------
workload | An application that the toolset has been configured to benchmark (Ex: linpack) | Y
ratio | The portion of the workload's resident set size that you want to keep in local memory, expressed as a decimal | Y
--id | The workload ID that's appended to the workload's name to create its container name. If let unset, it will default to 0 | N
--cpus | A comma separated list of CPUs to pin the workload to. If both this is left unset, the workload will be pinned to CPUs `[0, N-1]` where `N` is the number of CPUs listed in the workload's class | N

## Examples
### Linpack with 50% local memory on CPUs 4,5,6,7
    ./benchmark.py linpack 0.5 --cpus 4,5,6,7

### Quicksort with 30% local memory with an ID of 5
    ./benchmark.py quicksort 0.3 --id 5

## Adding Additional Workloads
New workloads can be added by modifying the workload_choices variable in `benchmark.py` and creating a new class for it in `lib/workloads.py`. 

# Multi-workload Benchmarks

## `server.py`
`server.py` runs on a separate (or even the same) machine from `scheduler.py`. Multiple `server.py` instances send execution-related data to a single `scheduler.py` instance, receiving workload execution directions in turn. `server.py` takes a single, optional flag, --log, that directs it to save a timestamped account of events to a file named `log.txt` in the same directory. 

## Potential Issues
We made a lot of assumptions about system configuration. `server.py` expects several files to exist on your system, mostly for sampling purposes. If they don't exist, we insert zeroes instead of reading their values.

## `scheduler.py`
This is the brains of the server-scheduler system. The scheduler is responsible for determining the arrival order of workloads, setting the shrinking policy, and aggregating all of the data from the server(s).

## Arguments
Argument            | Description | Required
--------------------------------|-------------------------------|--------------
seed | The seed used to initialize the randomized operations that the scheduler performs | Y
servers | A comma-separated list of ip:port combinations on which `server.py` instances are listening | Y
cpus | The number of cpus that each server is allowed to use | Y
mem | The amount of local memory that each server is allowed to use | Y
--remotemem, -r | Enables remote memory on each of the `server.py` instances | N
--max_far, -s | The maximum aggregate remote memory that servers are allowed to use. Enforced entirely in the scheduler. Default = Unlimited | N
--size | The total number of workloads to run. Default = 200| N
--workload | A comma-separated set of unique workloads to run. Default = quicksort,kmeans,memaslap | N
--ratios | A colon-separated set of ratios that correspond to the arguments for --workload. This determines how well-represented a particular workload type is in the aggregate. Default = 2:1:1 | N
--until | The maximum arrival time of a workload. Default = 20 | N
--uniform_ratio | Smallest local memory ratio for the uniform shrinking policy | N
--variable_ratios | A comma-separated list of minimum local memory ratios that correspond to the arguments for --workload | N
--start_burst | The number of workloads that will have their arrival time set to 0 instead of randomized. Default = 0 | N
--optimal | Use the optimal shrinking policy | N

## Examples

    ./scheduler.py 123 192.168.0.1:50051 8 8192 -r --max_far 4096 --size 100 \
    --workload quicksort,kmeans,linpack --ratios 3:1:1 --until 30 --optimal

Parameter            | Value | Explanation
--------------|-----------------|------
seed | 123 | Randomization seed. The same seed creates the same arrival pattern
servers | 192.168.0.1:50051 | Connect to a `server.py` instance at IP 192.168.0.1 that's listening on port 50051
cpus | 8 | The `server.py` instance can use a total of 8 CPUs
mem | 8192 (8192 = 8GB) | The `server.py` instance can use a total of 8GB of local memory
-r | Set | Enable the use of remote memory (for swapping)
--max_far | 4096 | The `server.py` instance can use a total of 4GB of remote memory
--size | 100 | A total of 100 workloads will be scheduled. The type/number are determined by `--workload` and `--ratios`
--workload | quicksort,kmeans,linpack | The previously-specified 100 workloads will consist of quicksort, kmeans, and linpack. The mixture is determined by `--ratios`
--ratios | 3:1:1 | The first, second, and third workloads in the comma-separated list passed to `--workload` constitute 60% (3/(3+1+1)), 20% (1/(3+1+1)), and 20% (1/(3+1+1)) of the 100 workloads respectively. In this example, there will be 60 quicksorts, 20 kmeans, and 20 linpacks scheduled.
--until | 30 | Each of the 30 workloads will have a random arrival time between 0 and 30 seconds
--optimal | Set | The `server.py` and `scheduler.py` will use the optimal shrinking policy. Setting this precludes using both `--uniform_ratio` and `--variable_ratios`

    ./scheduler.py 123 192.168.0.1:50051 8 8192 -r --size 100 --workload quicksort,kmeans,linpack \
    --ratios 3:1:1 --until 30 --variable_ratios 0.5,0.6,0.7

Parameter            | Value | Explanation
--------------|-----------------|------
seed | 123 | Randomization seed. The same seed creates the same arrival pattern
servers | 192.168.0.1:50051 | Connect to a `server.py` instance at IP 192.168.0.1 that's listening on port 50051
cpus | 8 | The `server.py` instance can use a total of 8 CPUs
mem | 8192 (8192 = 8GB) | The `server.py` instance can use a total of 8GB of local memory
-r | Set | Enable the use of remote memory (for swapping)
--max_far | Unset | The `server.py` instance can use unlimited remote memory
--size | 100 | A total of 100 workloads will be scheduled. The type/number are determined by `--workload` and `--ratios`
--workload | quicksort,kmeans,linpack | The previously-specified 100 workloads will consist of quicksort, kmeans, and linpack. The mixture is determined by `--ratios`
--ratios | 3:1:1 | The first, second, and third workloads in the comma-separated list passed to `--workload` constitute 60% (3/(3+1+1)), 20% (1/(3+1+1)), and 20% (1/(3+1+1)) of the 100 workloads respectively. In this example, there will be 60 quicksorts, 20 kmeans, and 20 linpacks scheduled.
--until | 30 | Each of the 30 workloads will have a random arrival time between 0 and 30 seconds
--variable_ratios | 0.5,0.6,0.7 | The three workloads (quicksort, kmeans, and linpack) will have their minimum ratios set to 0.5, 0.6, and 0.7 respectively. `server.py` and `scheduler.py` will use the variable shrinking policy. Setting this precludes using both `--uniform_ratio` and `--optimal`

    ./scheduler.py 123 192.168.0.1:50051,192.168.0.2:50051 8 8192 -r --size 250 \
    --workload quicksort,kmeans,linpack --ratios 3:1:1 --uniform_ratio 0.5 \
    --until 30 --start_burst 2

Parameter            | Value | Explanation
--------------|-----------------|------
seed | 123 | Randomization seed. The same seed creates the same arrival pattern
servers | 192.168.0.1:50051,192.168.0.2:50051 | Connect to `server.py` instances at IPs 192.168.0.1 and 192.168.0.2 that are both listening on port 50051
cpus | 8 | Each `server.py` instance can use a total of 8 CPUs
mem | 8192 (8192 = 8GB) | Each `server.py` instance can use a total of 8GB of local memory
-r | Set | Enable the use of remote memory (for swapping)
--max_far | Unset | Each `server.py` instance can use unlimited remote memory
--size | 250 | A total of 250 workloads will be scheduled. The type/number are determined by `--workload` and `--ratios`
--workload | quicksort,kmeans,linpack | The previously-specified 250 workloads will consist of quicksort, kmeans, and linpack. The mixture is determined by `--ratios`
--ratios | 3:1:1 | The first, second, and third workloads in the comma-separated list passed to `--workload` constitute 60% (3/(3+1+1)), 20% (1/(3+1+1)), and 20% (1/(3+1+1)) of the 100 workloads respectively. In this example, there will be 150 quicksorts, 50 kmeans, and 50 linpacks scheduled.
--uniform_ratio | 0.5 | The three workloads (quicksort, kmeans, and linpack) will have their minimum ratios set to 0.5. `server.py` and `scheduler.py` will use the uniform shrinking policy. Setting this precludes using both `--optimal` and `--variable_ratios`
--until | 30 | Each of the 30 workloads will have a random arrival time between 0 and 30 seconds
--start_burst | 2 | The first 2 workloads in the schedule will have their arrival times modified to be 0. This causes them to arrive immediately. 

## Further reading
For more information, please refer to our [paper](https://dl.acm.org/doi/abs/10.1145/3342195.3387522) accepted at [EUROSYS 2020](https://www.eurosys2020.org/)

## Questions
For additional questions please contact us at cfm@lists.eecs.berkeley.edu
