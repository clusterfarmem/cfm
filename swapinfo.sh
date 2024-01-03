for pid in $(ls /proc | grep '^[0-9]'); do
    awk '/Swap:/ {sum+=$2} END {print sum, "'$pid'"}' /proc/$pid/smaps 2>/dev/null
done | sort -nr | head -n 20