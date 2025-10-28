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

3. Install **Node** via Volta:

   ```bash
   volta install node
   ```

4. Pin the Node version to your project:

   ```bash
   volta pin node
   ```

5. Install project dependencies from the Bun lockfile:

   ```bash
   bun install
   ```

## Installation

1.  Navigate to the `pyodide-react` directory:
    ```bash
    cd pyodide-react
    ```
2.  Install the dependencies using Bun:
    ```bash
    bun install
    ```

## Testing the Application

Install dev prerequisites:

*  Install Python dev dependencies:
    ```bash
    pip install -r ../next/requirements.dev.txt
    ```

*  Install Playwright and dependencies:
    ```bash
    playwright install && playwright install-deps
    ```

Then run the following command from the `pyodide-react` directory:

```bash
./run_and_log.sh
```

This will build it, run an end-to-end test, and dump the log into `./debug_log.txt`

## Using the Application

Run the following command from the `pyodide-react` directory:

```bash
bun run index.tsx
```

This will build it and open the application in your browser.

## Specification

This project provides a web service for converting `draw.io` diagrams, which represent ontological models, into RML (RDF Mapping Language) files. This is achieved by leveraging a combination of a Bun-based web server, a Pyodide runtime for executing Python code, and a set of core Python scripts that perform the heavy lifting of the conversion.

### Components

1.  **React Front-End (`pyodide-react/app.tsx`)**:
    *   A Single Page Application (SPA) built with [React](https://react.dev/).
    *   It provides a user interface for uploading a `.drawio` file, a `.csv` file, and for providing ontology IRI prefixes.
    *   The entire application logic runs in the user's browser.

2.  **Pyodide Runtime**:
    *   [Pyodide](https://pyodide.org/) is a port of CPython to WebAssembly, which allows running Python code in the browser.
    *   The application loads Pyodide and the necessary Python packages on the fly.
    *   It creates a virtual file system within Pyodide and loads the Python scripts and user-provided files into it.

3.  **Python Conversion Scripts**:
    *   **`draw_io_parser.py`**: This script is responsible for parsing the `.drawio` file. It reads the XML structure of the diagram and extracts the nodes and edges, interpreting them as ontological individuals and relationships. It has robust error handling and options for sanitizing the identifiers to make them compliant with OWL IRI standards, including handling of spaces and other metacharacters. The output of this script is an `rdflib` graph that represents the structure of the `drawio` diagram.
    *   **`map_schema.py`**: This script is the RML generation engine. It takes the RDF graph produced by `draw_io_parser.py` and a CSV file as input, and it generates the final RML mapping file. It contains the logic for creating RML `TriplesMap`s, defining `subjectMap`s and `predicateObjectMap`s, and handling various RML and R2RML vocabulary terms. It includes functionality for preprocessing the source CSV file, disaggregating data, and handling complex mapping scenarios with conditional logic using FnO (Function Ontology). It appears to be a custom implementation of an RML generator that is tailored to the specific needs of this project.
    *   **Dependencies**: The Python environment within Pyodide is configured to install several libraries using `micropip`, including:
        *   `pandas`: Used for data manipulation, particularly for reading and processing the input CSV file.
        *   `rdflib`: A fundamental library for working with RDF in Python. It is used extensively by both `draw_io_parser.py` and `map_schema.py` to create, manipulate, and serialize RDF graphs.
        *   `requests`: A standard library for making HTTP requests. While not directly used in the main conversion workflow, it is loaded into the Pyodide environment.
        *   `lxml`: A high-performance library for processing XML and HTML. It is used by `draw_io_parser.py` to efficiently parse the `.drawio` file's underlying XML structure.

### Workflow

1.  The user opens the application in their browser.
2.  The React application loads, and in the background, it fetches and initializes the Pyodide runtime and the required Python packages.
3.  The user selects a `.drawio` file and a `.csv` file using the file inputs, and provides the ontology prefixes.
4.  The user clicks the "Convert" button.
5.  The application reads the content of the selected files.
6.  It loads the Python scripts (`draw_io_parser.py`, `map_schema.py`) and the user-provided files into Pyodide's virtual file system.
7.  It executes a Python script wrapper that:
    a.  Calls the `draw_io_parser.parse_drawio_content_to_graph` function to parse the `.drawio` file into an `rdflib` graph.
    b.  Invokes the `map_schema` script with this graph and the CSV data to generate the RML mapping.
8.  The resulting RML content is displayed on the page.
