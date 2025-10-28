#!/bin/bash

# Log the entire session to debug_log.txt
script -c "
    echo '--- Log Start: ' \$(date) '---'
    echo
    echo '--- System Details ---'
    uname -a
    echo
    echo '--- Bun Version ---'
    bun --version
    echo
    echo '--- Node Version ---'
    node --version
    echo
    echo '--- Installing Dependencies ---'
    bun install
    echo
    echo '--- Starting Server ---'
    bun run index.html &
    SERVER_PID=\$!
    echo 'Server started with PID: ' \$SERVER_PID
    sleep 5 # Give the server some time to start
    echo
    echo '--- Running E2E Test ---'
    # Update the playwright script to use port 8000
    #sed -i 's/localhost:3000/localhost:8000/g' ./verify_conversion.py
    pytest verify_conversion.py
    echo
    echo '--- Stopping Server ---'
    kill \$SERVER_PID
    echo
    echo '--- Log End: ' \$(date) '---'
" debug_log.txt
