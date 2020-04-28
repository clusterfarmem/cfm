#! /bin/bash

# the workload uses 1.5G of memory
set -e

echo "running linpack without memory limit"
sudo /usr/bin/time -v ./runme_xeon64

echo "running linpack 1152M"
../scripts/changemem_cgroup2.sh 1152M
sudo /usr/bin/time -v ../scripts/exec_cgroupv2.sh ./runme_xeon64

echo "running linpack 768M"
../scripts/changemem_cgroup2.sh 768M
sudo /usr/bin/time -v ../scripts/exec_cgroupv2.sh ./runme_xeon64

echo "running linpack 384M"
../scripts/changemem_cgroup2.sh 384M
sudo /usr/bin/time -v ../scripts/exec_cgroupv2.sh ./runme_xeon64
