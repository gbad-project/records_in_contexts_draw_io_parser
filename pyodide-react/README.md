# Pyodide React Runner

This is a client-side React application that uses Pyodide to run a Python script for converting `.drawio` diagrams into RML (RDF Mapping Language).

## Prerequisites

1. Install **Bun**:

   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```

2. Install **Volta**:

   ```bash
   curl https://get.volta.sh | bash
   ```

3. Install **Node** via Volta and pin it:

   ```bash
   volta install node
   volta pin node
   ```

4. Install project dependencies from the Bun lockfile:

   ```bash
   bun install
   ```

## Installation

1. Navigate to the `pyodide-react` directory:

   ```bash
   cd pyodide-react
   ```

2. Install the dependencies using Bun:

   ```bash
   bun install
   ```

## Testing the Application

1. Install Python dev dependencies:

   ```bash
   pip install -r ../next/requirements.dev.txt
   ```

2. Install Playwright and its system dependencies:

   ```bash
   playwright install && playwright install-deps
   ```

3. Run the demo script from the `pyodide-react` directory:

   ```bash
   ./run_demo.sh
   ```

   This prints system information, starts the development server, runs an end-to-end Playwright test, and saves the output to `run_demo.log`.

## Using the Application

Run the following command from the `pyodide-react` directory to build and launch the application in your browser:

```bash
bun run index.tsx
```

## Specification

This project provides a web service for converting `draw.io` diagrams, which represent ontological models, into RML (RDF Mapping Language) files. This is achieved by leveraging a combination of a Bun-based web server, a Pyodide runtime for executing Python code, and a set of core Python scripts that perform the heavy lifting of the conversion.

### Components

1. **React Front-End (`pyodide-react/app.tsx`)**:
   * A Single Page Application (SPA) built with [React](https://react.dev/).
   * It provides a user interface for uploading a `.drawio` file, a `.csv` file, and for providing ontology IRI prefixes.
   * The entire application logic runs in the user's browser.

2. **Pyodide Runtime**:
   * [Pyodide](https://pyodide.org/) is a port of CPython to WebAssembly, which allows running Python code in the browser.
   * The application loads Pyodide and the necessary Python packages on the fly.
   * It creates a virtual file system within Pyodide and loads the Python scripts and user-provided files into it.

3. **Python Conversion Scripts**:
   * **`draw_io_parser.py`**: Parses the `.drawio` file, extracting nodes and edges and interpreting them as ontological individuals and relationships. It can sanitize identifiers for OWL IRI compliance. Outputs an `rdflib` graph representing the diagram structure.
   * **`map_schema.py`**: Generates the final RML mapping file from the RDF graph and a CSV input. Handles creation of RML `TriplesMap`s and complex mapping scenarios with conditional logic using FnO (Function Ontology).
   * **Dependencies**: The Pyodide environment installs packages such as `pandas`, `rdflib`, `requests`, and `lxml` using `micropip`.

### Workflow

1. The user opens the application in their browser.
2. The React application loads, and in the background, it initializes Pyodide and the required Python packages.
3. The user selects a `.drawio` file and a `.csv` file using the file inputs, and provides the ontology prefixes.
4. The user clicks the "Convert" button.
5. The application reads the content of the selected files.
6. It loads the Python scripts (`draw_io_parser.py`, `map_schema.py`) and the user-provided files into Pyodide's virtual file system.
7. It executes a Python script wrapper that:
   a. Calls the `draw_io_parser.parse_drawio_content_to_graph` function to parse the `.drawio` file into an `rdflib` graph.
   b. Invokes the `map_schema` script with this graph and the CSV data to generate the RML mapping.
8. The resulting RML content is displayed on the page.
