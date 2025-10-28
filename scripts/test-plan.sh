#!/usr/bin/env bash
set -e

# generate timestamp
timestamp=$(date -u +%Y%m%d%H%M%S)
log_dir="logs"
mkdir -p "$log_dir"
log_file="$log_dir/architect-log-$timestamp.log"

{
  echo "start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  # run available tests; allow failure but record exit code
  if pytest >/tmp/architect-pytest.log 2>&1; then
    cat /tmp/architect-pytest.log
    echo "pytest_exit_code: 0"
  else
    cat /tmp/architect-pytest.log
    echo "pytest_exit_code: $?"
  fi
  echo "end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
} | tee "$log_file"
