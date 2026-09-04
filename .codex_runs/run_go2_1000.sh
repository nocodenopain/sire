#!/usr/bin/env bash
set -u

run_dir="$1"
mkdir -p "$run_dir"

train_log="$run_dir/train.log"
resource_log="$run_dir/resources.csv"
time_log="$run_dir/time.txt"
exit_file="$run_dir/exit_code"

cd /mnt/data/workspace/sire || exit 125
export PYTHONPATH="python/src:demo/demo_python:/home/lqf/code/rsl_rl"

printf 'timestamp,elapsed_seconds,train_alive,rss_kib,system_used_mib,system_available_mib,log_bytes\n' > "$resource_log"

/usr/bin/time -v -o "$time_log" .venv/bin/python -u \
  demo/demo_python/SireRLGym/scripts/train.py \
  --task go2 \
  --num_envs 1024 \
  --sire_batch_threads 16 \
  --max_iterations 1000 \
  --save_interval 50 \
  --flat_terrain > "$train_log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$run_dir/train.pid"

started=$(date +%s)
two_hour_written=0
while kill -0 "$train_pid" 2>/dev/null; do
  now=$(date +%s)
  elapsed=$((now - started))
  rss_kib=$(ps -o rss= -p "$train_pid" --ppid "$train_pid" 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  system_used_mib=$(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {print int((total-available)/1024)}' /proc/meminfo)
  system_available_mib=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  log_bytes=$(stat -c %s "$train_log" 2>/dev/null || printf '0')
  printf '%s,%s,1,%s,%s,%s,%s\n' "$(date --iso-8601=seconds)" "$elapsed" "$rss_kib" "$system_used_mib" "$system_available_mib" "$log_bytes" >> "$resource_log"
  if (( elapsed >= 7200 && two_hour_written == 0 )); then
    printf 'alive at %s after %s seconds\n' "$(date --iso-8601=seconds)" "$elapsed" > "$run_dir/two_hour_check.txt"
    two_hour_written=1
  fi
  sleep 300
done

wait "$train_pid"
status=$?
printf '%s\n' "$status" > "$exit_file"
printf '%s,%s,0,0,%s,%s,%s\n' \
  "$(date --iso-8601=seconds)" "$(( $(date +%s) - started ))" \
  "$(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {print int((total-available)/1024)}' /proc/meminfo)" \
  "$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)" \
  "$(stat -c %s "$train_log" 2>/dev/null || printf '0')" >> "$resource_log"
exit "$status"
