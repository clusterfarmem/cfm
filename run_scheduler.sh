sudo python scheduler.py 123 10.10.1.2:50051,10.10.1.3:50051,10.10.1.4:50051,10.10.1.5:50051,10.10.1.6:50051,10.10.1.7:50051 \
32 65536 -r --size 200 --max_far 196608 \
--workload quicksort --ratios 1 --uniform_ratio 0.5 \
--until 15 