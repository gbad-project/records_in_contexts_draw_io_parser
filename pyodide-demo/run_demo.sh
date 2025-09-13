#!/bin/bash
set -e

echo "--- System Information ---"
uname -a
echo "--------------------------"

echo "--- Starting Demo ---"
echo "Start time: $(date)"

npx playwright test > run_demo.log 2>&1

echo "--- Demo Finished ---"
echo "End time: $(date)"
echo "Log file created at run_demo.log"
