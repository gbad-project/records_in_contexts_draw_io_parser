#!/usr/bin/env bash
set -e
PYODIDE_VERSION=0.23.4
TARGET_DIR="pyodide"
if [ -f "$TARGET_DIR/pyodide.js" ]; then
  echo "Pyodide already downloaded"
  exit 0
fi
mkdir -p "$TARGET_DIR"
ARCHIVE="pyodide-$PYODIDE_VERSION.tar.bz2"
URL="https://github.com/pyodide/pyodide/releases/download/$PYODIDE_VERSION/$ARCHIVE"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
curl -L "$URL" -o "$TMPDIR/$ARCHIVE"
tar -xjf "$TMPDIR/$ARCHIVE" -C "$TARGET_DIR" --strip-components=1
