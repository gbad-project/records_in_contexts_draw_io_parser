#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exec &> >(tee -a "$SCRIPT_DIR/run_demo.log")

set -ex

echo "--- System Information ---"
uname -a
echo "--------------------------"

echo "--- Starting Demo ---"
echo "Start time: $(date)"

cd "$SCRIPT_DIR"
npx playwright test

echo "--- Demo Finished ---"
echo "End time: $(date)"
echo "Log file created at $SCRIPT_DIR/run_demo.log"
