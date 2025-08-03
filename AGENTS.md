# GBAD Refactoring Plan

This document outlines the refactoring plan for the GBAD project, organized as a series of tickets in a backlog.

## In Progress

-   **TICKET-1:** Refactor Global Variables from `draw_io_parser.py` ([Details](tickets/TICKET-1.xml))
    -   **Description:** Move hardcoded global variables (`BASE_URI`, `_prefixes`, `_classes`, `_object_properties`, `_datatype_properties`, `DEFAULT_...`, `OWL_METACHARACTERS`) to the new `gbad_core/config.py` and `gbad_core/ric_ontology.py` modules.
    -   **Assignee:** Agent
    -   **Status:** Planned

## Backlog

### `draw_io_parser.py`

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Global Variables from draw_io_parser.py</skos:note></rdf:Description> -->

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Preserve Custom Exceptions from draw_io_parser.py</skos:note></rdf:Description> -->
-   **TICKET-2:** Preserve Custom Exceptions from `draw_io_parser.py` ([Details](tickets/TICKET-2.xml))
    -   **Description:** Move the custom exception classes (`NothingToParseException`, `NotInKnownException`, etc.) to the `gbad_core/drawio.py` module.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Data Classes from draw_io_parser.py</skos:note></rdf:Description> -->
-   **TICKET-3:** Refactor Data Classes from `draw_io_parser.py` ([Details](tickets/TICKET-3.xml))
    -   **Description:** Move the `Individual` and `Arrow` data classes to `gbad_core/drawio.py` to be used as the structured representation of the parsed graph. The `SerialisationConfig` class will be replaced by the new configuration object from `gbad_core/config.py`.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor NodeHTMLParser from draw_io_parser.py</skos:note></rdf:Description> -->
-   **TICKET-4:** Refactor `NodeHTMLParser` from `draw_io_parser.py` ([Details](tickets/TICKET-4.xml))
    -   **Description:** Move the `NodeHTMLParser` class to `gbad_core/drawio.py` as a private helper class for the new `DrawIOParser`.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor DrawIOXMLTree into DrawIOParser</skos:note></rdf:Description> -->
-   **TICKET-5:** Refactor `DrawIOXMLTree` into `DrawIOParser` ([Details](tickets/TICKET-5.xml))
    -   **Description:** Create the new `DrawIOParser` class in `gbad_core/drawio.py` and move the core parsing logic from `DrawIOXMLTree` into it. The new `parse()` method should return a structured representation of the graph using the `Individual` and `Arrow` data classes.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Define Behavior for Ambiguous Node Structures</skos:note></rdf:Description> -->
-   **TICKET-6:** Define Behavior for Ambiguous Node Structures ([Details](tickets/TICKET-6.xml))
    -   **Description:** The current parser assumes a specific structure for individual nodes. The human staff needs to decide how the system should handle nodes that do not follow this structure.
    -   **Assignee:** Human
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Define Handling for Non-RiC-O Properties</skos:note></rdf:Description> -->
-   **TICKET-7:** Define Handling for Non-RiC-O Properties ([Details](tickets/TICKET-7.xml))
    -   **Description:** The current parser can be configured to allow arrows with labels that are not known RiC-O properties. The human staff needs to decide how these non-standard properties should be handled in the refactored pipeline.
    -   **Assignee:** Human
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Identifier Cleaning Functions from draw_io_parser.py</skos:note></rdf:Description> -->
-   **TICKET-8:** Refactor Identifier Cleaning Functions from `draw_io_parser.py` ([Details](tickets/TICKET-8.xml))
    -   **Description:** Move the identifier cleaning functions (`_handle_spaces`, `_replace_metacharacter`, `_replace_metacharacters`) to the `DrawIOParser` class and make them more configurable.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Remove Serialization Logic from draw_io_parser.py</skos:note></rdf:Description> -->
-   **TICKET-9:** Remove Serialization Logic from `draw_io_parser.py` ([Details](tickets/TICKET-9.xml))
    -   **Description:** The serialization functions (`_infer_type`, `_serialise_facts`, `_serialise_block`, `_preamble`, `serialise`) will be removed from the refactored `drawio` module. The `DrawIOParser` will only parse, not serialize.
    -   **Assignee:** Agent
    -   **Status:** Backlog

### `map_schema.py`

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Global Variables from map_schema.py</skos:note></rdf:Description> -->
-   **TICKET-10:** Refactor Global Variables from `map_schema.py` ([Details](tickets/TICKET-10.xml))
    -   **Description:** Move hardcoded global variables (`BASE_URI`, CSV column names, placeholder masks, `rico_authtp_dict`, etc.) to the new `gbad_core/config.py` module.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Preprocessing Logic from map_schema.py</skos:note></rdf:Description> -->
-   **TICKET-11:** Refactor Preprocessing Logic from `map_schema.py` ([Details](tickets/TICKET-11.xml))
    -   **Description:** Move the `add_preprocess` and `auth_preprocess` functions into a new `CSVPreprocessor` class in `gbad_core/preprocessing.py`. The preprocessing steps should be made generic and configurable.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Provide Rationale for Preprocessing Logic</skos:note></rdf:Description> -->
-   **TICKET-12:** Provide Rationale for Preprocessing Logic ([Details](tickets/TICKET-12.xml))
    -   **Description:** The human staff needs to provide a detailed explanation of the business rules and data semantics for the preprocessing steps.
    -   **Assignee:** Human
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules_google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor RML Generation Logic from map_schema.py</skos:note></rdf:Description> -->
-   **TICKET-13:** Refactor RML Generation Logic from `map_schema.py` ([Details](tickets/TICKET-13.xml))
    -   **Description:** Create a new `RMLGenerator` class in `gbad_core/rml.py` to handle the core RML generation logic. This includes generalizing the placeholder handling mechanism (`disaggregate_rico_authtp`, `disaggregate_refd_file`) and preserving the FnO mapping patterns.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Define Requirements for Disaggregation Logic</skos:note></rdf:Description> -->
-   **TICKET-14:** Define Requirements for Disaggregation Logic ([Details](tickets/TICKET-14.xml))
    -   **Description:** The human staff needs to provide a clear definition of the different entity types and reference schemes that the system needs to support to generalize the disaggregation logic.
    -   **Assignee:** Human
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Identify Additional Conditional Mapping Scenarios</skos:note></rdf:Description> -->
-   **TICKET-15:** Identify Additional Conditional Mapping Scenarios ([Details](tickets/TICKET-15.xml))
    -   **Description:** The human staff needs to provide a list of any additional conditional mapping scenarios that the refactored system should support.
    -   **Assignee:** Human
    -   **Status:** Backlog

### `map_rml.py`

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor RML Execution Logic from map_rml.py</skos:note></rdf:Description> -->
-   **TICKET-16:** Refactor RML Execution Logic from `map_rml.py` ([Details](tickets/TICKET-16.xml))
    -   **Description:** Move the RML mapping execution logic into a new `RMLMapper` class in `gbad_core/mapper.py`. The class should be independent of the specific RML mapper implementation.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Post-processing Logic from map_rml.py</skos:note></rdf:Description> -->
-   **TICKET-17:** Refactor Post-processing Logic from `map_rml.py` ([Details](tickets/TICKET-17.xml))
    -   **Description:** Move the post-processing logic into a new `gbad_core/postprocessing.py` module, with configurable cleanup steps.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Provide Rationale for Post-processing Rules</skos:note></rdf:Description> -->
-   **TICKET-18:** Provide Rationale for Post-processing Rules ([Details](tickets/TICKET-18.xml))
    -   **Description:** The human staff needs to provide a list of all known data quality issues and the desired resolution for each, to inform the configuration of the post-processing module.
    -   **Assignee:** Human
    -   **Status:** Backlog

### `merge_graphs.py`

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Refactor Graph Merging Logic from merge_graphs.py</skos:note></rdf:Description> -->
-   **TICKET-19:** Refactor Graph Merging Logic from `merge_graphs.py` ([Details](tickets/TICKET-19.xml))
    -   **Description:** Move the graph merging logic into a new `GraphMerger` class in `gbad_core/graph_merger.py`. The class should be able to merge a list of RDF graph files or `rdflib.Graph` objects into a single `rdflib.Dataset`.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-ASK"><assignee>human</assignee><skos:note>Define Provenance Requirements</skos:note></rdf:Description> -->
-   **TICKET-20:** Define Provenance Requirements ([Details](tickets/TICKET-20.xml))
    -   **Description:** The human staff needs to provide a list of all metadata that should be included in the named graph for provenance.
    -   **Assignee:** Human
    -   **Status:** Backlog

### Infrastructure and Testing

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Create gbad_core Package</skos:note></rdf:Description> -->
-   **TICKET-21:** Create `gbad_core` Package ([Details](tickets/TICKET-21.xml))
    -   **Description:** Create the `gbad_core` directory and `__init__.py` file.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Create Pipeline Facade</skos:note></rdf:Description> -->
-   **TICKET-22:** Create Pipeline Facade ([Details](tickets/TICKET-22.xml))
    -   **Description:** Create the `Pipeline` class in `gbad_core/pipeline.py` to orchestrate the entire conversion process.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Create Configuration Module</skos:note></rdf:Description> -->
-   **TICKET-23:** Create Configuration Module ([Details](tickets/TICKET-23.xml))
    -   **Description:** Create the `gbad_core/config.py` module to manage all pipeline configuration.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Create CLI and Streamlit App</skos:note></rdf:Description> -->
-   **TICKET-24:** Create CLI and Streamlit App ([Details](tickets/TICKET-24.xml))
    -   **Description:** Create the `main.py` and `streamlit_app.py` files to provide a command-line interface and a user-facing web application.
    -   **Assignee:** Agent
    -   **Status:** Backlog

<!-- <rdf:Description rdf:about="tag:jules@google.com,2025-08-02:AICODE-TODO"><assignee>agent</assignee><skos:note>Implement Testing Pipeline</skos:note></rdf:Description> -->
-   **TICKET-25:** Implement Testing Pipeline ([Details](tickets/TICKET-25.xml))
    -   **Description:** Create the necessary directory structure for tests (`tests/unit`, `tests/integration`, `tests/data`) and implement the unit, integration, and regression tests for the `gbad_core` library.
    -   **Assignee:** Agent
    -   **Status:** Backlog

## Completed

(no tickets completed)
