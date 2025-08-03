# GBAD Refactoring Plan

## 1. Project Goal

The primary goal of this project is to refactor a collection of Python scripts into a clean, modular, and extensible pipeline for converting Draw.io diagrams and CSV files into N-Quads. This new pipeline will be designed to be highly configurable and free of hardcoded logic, enabling its use in a variety of scenarios, including a user-friendly Streamlit application and as a library for developers.

The final pipeline will support the following workflow:
- **Input:** Multiple Draw.io diagrams and multiple CSV files.
- **Configuration:** A flexible configuration mechanism to define the mapping and transformation rules.
- **Processing:** A series of well-defined steps to parse the diagrams, preprocess the CSVs, generate and execute RML mappings, and merge the resulting RDF graphs.
- **Output:** A single, consolidated N-Quads file.

## 2. Current Architecture and Problems

The existing solution is comprised of four main Python scripts that form a sequential pipeline: `draw_io_parser.py`, `map_schema.py`, `map_rml.py`, and `merge_graphs.py`.

### Current Pipeline:
1.  A Draw.io diagram is manually created to represent the mapping logic.
2.  `draw_io_parser.py` converts the diagram into a base TTL/OWL file.
3.  `map_schema.py` uses this TTL file and a source CSV to generate an RML mapping file.
4.  `map_rml.py` executes the RML mapping to produce a TTL file.
5.  `merge_graphs.py` combines multiple TTL files into a single N-Quads dataset.

### Key Problems:
- **Hardcoding:** The scripts are replete with hardcoded file paths, schema-specific logic (e.g., for `add` and `auth` schemas), and data cleaning steps. This makes the pipeline inflexible and difficult to adapt to new use cases.
- **Tight Coupling:** The scripts are tightly coupled, with implicit dependencies and shared logic that is not clearly defined. This makes it difficult to modify or extend any part of the pipeline without affecting other parts.
- **Lack of Modularity:** The logic is not well-encapsulated, making it hard to reuse or test individual components.
- **Command-Line Interface:** The entire pipeline is designed to be run from the command line, which is not suitable for integration into a web application or for use as a library.

## 3. Proposed Architecture

To address these issues, we will refactor the existing codebase into a new Python package named `gbad_core`. This package will feature a modular architecture with a clear separation of concerns, making it easy to configure, extend, and maintain.

The new architecture will be composed of the following main components:
- **`gbad_core` package:** The root package for all the refactored code.
- **`drawio` module:** For parsing Draw.io diagrams.
- **`rml` module:** For generating RML mappings.
- **`preprocessing` module:** For configurable CSV preprocessing.
- **`mapper` module:** For executing RML mappings.
- **`postprocessing` module:** For configurable RDF graph cleanup.
- **`graph_merger` module:** For merging multiple RDF graphs.
- **`pipeline` module:** A high-level facade to orchestrate the entire pipeline.
- **`config` module:** For managing all pipeline configurations.

## 4. Refactoring Tasks

The following tasks will be executed by junior agents to implement the new architecture.

### Task 1: Create the `gbad_core` package
- **Action:** Create a new directory named `gbad_core` in the root of the repository.
- **Action:** Add an empty `__init__.py` file to the `gbad_core` directory to mark it as a Python package.

### Task 2: Refactor `draw_io_parser.py`
- **Action:** Create a new module `gbad_core/drawio.py`.
- **Action:** Implement a `DrawIOParser` class in `gbad_core/drawio.py` that encapsulates the core parsing logic from `draw_io_parser.py`.
- **Details:**
    - The `__init__` method should accept the Draw.io XML content as a string.
    - The `parse()` method should return a structured representation of the graph (e.g., a list of node and edge objects) instead of the current OWL Manchester string.
    - The hardcoded RiC-O terms should be moved to a separate, configurable module (`gbad_core/ric_ontology.py`).

### Task 3: Refactor `map_schema.py`
- **Action:** Create a new module `gbad_core/rml.py`.
- **Action:** Implement an `RMLGenerator` class in `gbad_core/rml.py` to generate RML mappings.
- **Details:**
    - The `RMLGenerator` class will take the parsed Draw.io graph from `DrawIOParser` and a configuration object as input.
    - The logic for handling placeholders like `{RICO_AUTHTP}` will be generalized and made configurable.
- **Action:** Create a new module `gbad_core/preprocessing.py`.
- **Action:** Implement a `CSVPreprocessor` class in `gbad_core/preprocessing.py` to handle CSV preprocessing.
- **Details:**
    - The `CSVPreprocessor` class will be configurable with a series of preprocessing steps (e.g., column splitting, data correction).
    - The configuration for these steps will be externalized and not hardcoded.

### Task 4: Refactor `map_rml.py`
- **Action:** Create a new module `gbad_core/mapper.py`.
- **Action:** Implement an `RMLMapper` class in `gbad_core/mapper.py` to execute RML mappings.
- **Details:**
    - The `RMLMapper` class will take an RML mapping and a source CSV as input.
    - It will be designed to be independent of the specific RML mapper implementation (e.g., it will support the existing JAR file, but could be extended to use other mappers).
- **Action:** Create a new module `gbad_core/postprocessing.py`.
- **Action:** Move the RDF graph cleanup logic to `gbad_core/postprocessing.py`, with configurable cleanup steps.

### Task 5: Refactor `merge_graphs.py`
- **Action:** Create a new module `gbad_core/graph_merger.py`.
- **Action:** Implement a `GraphMerger` class in `gbad_core/graph_merger.py` to merge RDF graphs.
- **Details:**
    - The `GraphMerger` class will accept a list of RDF graph files or `rdflib.Graph` objects.
    - It will merge them into a single `rdflib.Dataset`, creating a named graph for each input for provenance.

### Task 6: Create a Pipeline Facade
- **Action:** Create a new module `gbad_core/pipeline.py`.
- **Action:** Implement a `Pipeline` class in `gbad_core/pipeline.py` to orchestrate the entire conversion process.
- **Details:**
    - The `Pipeline` class will provide a simple, high-level interface, such as a `run(drawio_files, csv_files, config)` method.
    - This will serve as the main entry point for the Streamlit application and for developers using the library.

### Task 7: Create a Configuration Module
- **Action:** Create a new module `gbad_core/config.py`.
- **Action:** This module will manage all configuration for the pipeline, including RiC-O ontology terms, preprocessing steps, post-processing steps, and paths to external tools.

### Task 8: Create a CLI and a Streamlit App
- **Action:** Create a new `main.py` file in the root of the repository that uses the `gbad_core` library to provide a command-line interface.
- **Action:** Create a new `streamlit_app.py` file in the root of the repository that uses the `gbad_core` library to build the user-facing web application.

This plan provides a clear path to refactoring the existing codebase into a robust and flexible pipeline. Junior agents should follow these tasks sequentially to ensure a smooth and successful implementation.

## 5. Testing Strategy

A comprehensive testing strategy is crucial to ensure the quality and correctness of the refactored pipeline. The following testing layers will be implemented:

### Unit Tests
- **Goal:** To verify the functionality of individual components (classes and methods) in isolation.
- **Framework:** `pytest`
- **Implementation:** Each module in the `gbad_core` package will have a corresponding test module (e.g., `tests/unit/test_drawio.py`). These tests will cover the public API of each component and will use mock objects and test data to isolate the component from its dependencies.

### Integration Tests
- **Goal:** To verify that the different components of the pipeline work together as expected.
- **Framework:** `pytest`
- **Implementation:** Integration tests will cover the entire pipeline, from parsing the Draw.io diagrams to generating the final N-Quads file. These tests will use a small, representative set of test data to exercise the full pipeline.

### Regression Tests
- **Goal:** To ensure that the refactored pipeline produces the same output as the original scripts for a given set of inputs.
- **Implementation:**
    1. A suite of test cases will be created, each consisting of a set of input Draw.io diagrams and CSV files.
    2. The original pipeline will be run on these test cases, and the output N-Quads files will be stored as "golden" reference files.
    3. The refactored pipeline will be run on the same test cases, and its output will be compared to the golden files. Any differences will be flagged as a regression.
    4. A helper script will be created to automate this comparison, ignoring irrelevant differences (e.g., in the order of triples).

### Test Data
- A `tests/data` directory will be created to store all necessary test data, including:
    - Sample Draw.io files.
    - Sample CSV files.
    - Expected output files for unit and integration tests.
    - "Golden" reference files for regression tests.

### Task 9: Implement the Testing Pipeline
- **Action:** Create the necessary directory structure for tests (e.g., `tests/unit`, `tests/integration`, `tests/data`).
- **Action:** Implement the unit tests for all modules in the `gbad_core` package.
- **Action:** Implement the integration tests for the full pipeline.
- **Action:** Implement the regression testing framework, including the generation of golden files and the comparison script.
