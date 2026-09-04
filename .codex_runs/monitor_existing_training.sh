#!/usr/bin/env bash
set -u

run_dir="$1"
train_pid=$(cat "$run_dir/train.pid")
resource_log="$run_dir/resources_v2.csv"
started=$(date +%s)

printf 'timestamp,monitor_elapsed_seconds,train_alive,rss_kib,system_used_mib,system_available_mib,latest_iteration,log_bytes\n' > "$resource_log"
while kill -0 "$train_pid" 2>/dev/null; do
  now=$(date +%s)
  rss_kib=$(ps -o rss= -p "$train_pid" --ppid "$train_pid" 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  system_used_mib=$(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {print int((total-available)/1024)}' /proc/meminfo)
  system_available_mib=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  latest_iteration=$(grep -a 'Learning iteration' "$run_dir/train.log" 2>/dev/null | tail -n 1 | sed -E 's/.*iteration ([0-9]+)\/1000.*/\1/' || true)
  log_bytes=$(stat -c %s "$run_dir/train.log" 2>/dev/null || printf '0')
  printf '%s,%s,1,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$((now-started))" "$rss_kib" \
    "$system_used_mib" "$system_available_mib" "$latest_iteration" "$log_bytes" >> "$resource_log"
  sleep 300
done

printf '%s,%s,0,0,%s,%s,,%s\n' \
  "$(date --iso-8601=seconds)" "$(( $(date +%s)-started ))" \
  "$(awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {available=$2} END {print int((total-available)/1024)}' /proc/meminfo)" \
  "$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)" \
  "$(stat -c %s "$run_dir/train.log" 2>/dev/null || printf '0')" >> "$resource_log"
