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
    echo '--- Building React Application ---'
    bun build ./app.tsx --outdir ./dist
    echo
    echo '--- Log End: ' \$(date) '---'
" debug_log.txt
