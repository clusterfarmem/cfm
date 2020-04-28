#! /bin/bash

# the workload uses 1.5G of memory
set -e
PERF="perf stat -d"

#echo "running linpack without memory limit"
#sudo $PERF ./runme_xeon64
#
#echo "running linpack 1152M"
#../scripts/changemem_cgroup2.sh 1152M
#sudo ../scripts/exec_cgroupv2.sh $PERF ./runme_xeon64
#
#echo "running linpack 768M"
#../scripts/changemem_cgroup2.sh 768M
#sudo ../scripts/exec_cgroupv2.sh $PERF ./runme_xeon64

echo "running linpack 384M"
../scripts/changemem_cgroup2.sh 384M
sudo ../scripts/exec_cgroupv2.sh ./runme_xeon64
