#!/bin/bash

# Check if file path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-nquads-file>"
    echo "Example: $0 /path/to/your/data.nq"
    exit 1
fi

FILE_PATH="$1"

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found!"
    exit 1
fi

echo "Uploading $FILE_PATH to ResearchSpace..."

curl -X POST \
  -H "Content-Type: application/n-quads" \
  --data-binary "@$FILE_PATH" \
  "http://localhost:10214/blazegraph/namespace/kb/rdf-graph-store"

echo ""
echo "Operation completed!"
