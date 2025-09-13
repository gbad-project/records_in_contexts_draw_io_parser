# Pyodide React Simplification Report

## Demo script

- Ran `./run_demo.sh` inside `pyodide-react`.
- Script failed because the Python `playwright` package was missing.
- Attempted to install requirements via `pip install -r ../next/requirements.dev.txt` and `playwright install && playwright install-deps`.
- The dependency installation started a large system package install which was interrupted due to resource limits, so the demo could not complete.
- Existing `run_demo.log` captures the initial attempts and remains in the repository.

## Simplified parser

- Created `pyodide-react/draw_io_parser_simple.py` implementing a minimal parser that reads a `.drawio` XML file, extracts vertex IDs and emits an RDF graph where each vertex is typed as `ex:Node`.
- Updated `pyodide-react/app.tsx` to load this simplified parser instead of the full conversion pipeline. The React app now uploads only a `.drawio` file and displays the Turtle serialisation returned by the parser.

## Files changed

- `pyodide-react/draw_io_parser_simple.py`
- `pyodide-react/app.tsx`
- `pyodide-react/run_demo.log` (generated/updated by demo attempt)
- `report.md`
