# Pyodide Runner

This is a web server that uses Pyodide to run a Python script for converting `.drawio` diagrams into RML (RDF Mapping Language).

## Prerequisites

*   [Bun](https://bun.sh/) must be installed.

## Installation

1.  Navigate to the `pyodide` directory:
    ```bash
    cd pyodide
    ```
2.  Install the dependencies using Bun:
    ```bash
    bun install
    ```

## Running the Server

To start the server, run the following command from the `pyodide` directory:

```bash
bun run app.ts
```

The server will start on `http://localhost:3000`.

## Usage

The server exposes a `/convert` endpoint that accepts `POST` requests with a `multipart/form-data` body.

### Request Body

The request body should contain the following parts:

*   `drawioFile`: The `.drawio` file to be converted.
*   `csvFile`: A CSV file used by the conversion script.
*   `ontologyIris`: A string containing the ontology IRI prefixes, one per line (e.g., `rico: https://...`).

### Example Request

You can use `curl` to send a request to the server:

```bash
curl -X POST \\
  -F "drawioFile=@path/to/your/file.drawio" \\
  -F "csvFile=@path/to/your/file.csv" \\
  -F "ontologyIris=rico: https://www.ica.org/standards/RiC/ontology#" \\
  http://localhost:3000/convert
```

The server will respond with the generated RML file.

## Specification

This project provides a web service for converting `draw.io` diagrams, which represent ontological models, into RML (RDF Mapping Language) files. This is achieved by leveraging a combination of a Bun-based web server, a Pyodide runtime for executing Python code, and a set of core Python scripts that perform the heavy lifting of the conversion.

### Components

1.  **Bun Web Server (`pyodide/app.ts`)**:
    *   A web server built using [Bun](https://bun.sh/), a fast JavaScript runtime.
    *   It exposes a single endpoint, `/convert`, which accepts `POST` requests.
    *   It handles `multipart/form-data` requests containing the `drawio` file, a CSV file, and ontology prefixes.
    *   It initializes a Pyodide environment and orchestrates the conversion process.

2.  **Pyodide Runtime**:
    *   [Pyodide](https://pyodide.org/) is a port of CPython to WebAssembly, which allows running Python code in a web environment (in this case, within the Bun server).
    *   The server uses Pyodide to execute the Python conversion scripts.
    *   It creates a virtual file system within Pyodide and loads the necessary Python scripts and data files into it.
    *   It passes data from the JavaScript environment to the Python environment.

3.  **Python Conversion Scripts**:
    *   **`draw_io_parser.py`**: This script is responsible for parsing the `.drawio` file. It reads the XML structure of the diagram and extracts the nodes and edges, interpreting them as ontological individuals and relationships. It has robust error handling and options for sanitizing the identifiers to make them compliant with OWL IRI standards, including handling of spaces and other metacharacters. The output of this script is an `rdflib` graph that represents the structure of the `drawio` diagram.
    *   **`map_schema.py`**: This script is the RML generation engine. It takes the RDF graph produced by `draw_io_parser.py` and a CSV file as input, and it generates the final RML mapping file. It contains the logic for creating RML `TriplesMap`s, defining `subjectMap`s and `predicateObjectMap`s, and handling various RML and R2RML vocabulary terms. It includes functionality for preprocessing the source CSV file, disaggregating data, and handling complex mapping scenarios with conditional logic using FnO (Function Ontology). It appears to be a custom implementation of an RML generator that is tailored to the specific needs of this project.
    *   **Dependencies**: The Python environment within Pyodide is configured to install several libraries using `micropip`, including:
        *   `pandas`: Used for data manipulation, particularly for reading and processing the input CSV file.
        *   `rdflib`: A fundamental library for working with RDF in Python. It is used extensively by both `draw_io_parser.py` and `map_schema.py` to create, manipulate, and serialize RDF graphs.
        *   `requests`: A standard library for making HTTP requests. While not directly used in the main conversion workflow, it is loaded into the Pyodide environment.
        *   `lxml`: A high-performance library for processing XML and HTML. It is used by `draw_io_parser.py` to efficiently parse the `.drawio` file's underlying XML structure.

### Workflow

1.  The Bun server receives a `POST` request at the `/convert` endpoint.
2.  It extracts the `drawio` file, CSV file, and ontology prefixes from the request body.
3.  It initializes the Pyodide environment and installs the required Python packages.
4.  It loads the Python conversion scripts and the user-provided files into Pyodide's virtual file system.
5.  It constructs and executes a Python script wrapper that orchestrates the conversion:
    a.  The `draw_io_parser.parse_drawio_content_to_graph` function is called to parse the `.drawio` file into an `rdflib` graph.
    b.  The `map_schema` script is then invoked with this graph and the CSV data to generate the RML mapping.
6.  The resulting RML content is returned as the response to the HTTP request.
