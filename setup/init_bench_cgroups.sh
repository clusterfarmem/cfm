#!/bin/bash

CGROUP_ROOT=/cgroup2
CGROUP_BENCH=$CGROUP_ROOT/benchmarks
USER=$(whoami)

echo "will setup cgroups at $CGROUP_ROOT for user $USER"
sudo mount -t cgroup2 nodev $CGROUP_ROOT
sudo sh -c "echo '+memory' > $CGROUP_ROOT/cgroup.subtree_control"

sudo mkdir $CGROUP_BENCH
sudo sh -c "echo '+memory' > $CGROUP_BENCH/cgroup.subtree_control"

sudo chown $USER -R $CGROUP_ROOT

echo "enabling readahead"
sudo sh -c "echo 3 > /proc/sys/vm/page-cluster"

echo "done"
