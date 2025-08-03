# Analysis Report for GBAD Refactoring Plan

This report details the analysis of the four key Python scripts that form the current GBAD data processing pipeline. It serves as a verification that the refactoring plan outlined in `AGENTS.md` is based on a thorough understanding of the existing codebase.

## `draw_io_parser.py` Analysis

This script is responsible for parsing Draw.io XML files and converting them into OWL Manchester syntax.

### Global Variables

-   **`BASE_URI`**: Hardcoded base URI.
    -   **`AGENTS.md` Reflection:** The refactoring plan specifies that all hardcoded values, including this one, will be moved to a central, configurable module (`gbad_core/config.py`).
-   **`_prefixes`, `_classes`, `_object_properties`, `_datatype_properties`**: Hardcoded lists of RiC-O terms.
    -   **`AGENTS.md` Reflection:** These will be moved to a dedicated ontology module (`gbad_core/ric_ontology.py`) to make them configurable and extensible, as outlined in Task 2 of the plan.
-   **Type Aliases (`Blocks`, `Cell`, etc.)**: These are for type hinting and will be preserved in the refactored code for clarity.
-   **Constants (`DEFAULT_...`, `OWL_METACHARACTERS`)**: Default values for script execution.
    -   **`AGENTS.md` Reflection:** These will also be moved to the configuration module (`gbad_core/config.py`).

### Exceptions

-   **Custom Exceptions (`NothingToParseException`, `NotInKnownException`, etc.)**: These are well-defined custom exceptions for handling parsing errors.
    -   **`AGENTS.md` Reflection:** These will be preserved in the refactored `gbad_core/drawio.py` module to ensure robust error handling.

### Data Classes

-   **`Individual`**, **`Arrow`**: These represent the core data structures of the parsed graph.
    -   **`AGENTS.md` Reflection:** Task 2 of the plan specifies that the `DrawIOParser` will return a structured representation of the graph. These data classes will be used for that purpose.
-   **`SerialisationConfig`**: Holds configuration for OWL serialization.
    -   **`AGENTS.md` Reflection:** This will be replaced by the central configuration object from `gbad_core/config.py`.

### Classes

-   **`NodeHTMLParser`**: A utility class for parsing HTML content within Draw.io nodes.
    -   **`AGENTS.md` Reflection:** This is a helper class for the parser and will be kept as a private class within the `gbad_core/drawio.py` module.
-   **`DrawIOXMLTree`**: The main class for parsing the Draw.io XML.
    -   **`AGENTS.md` Reflection:** The logic of this class will be moved into the new `DrawIOParser` class in `gbad_core/drawio.py`, as specified in Task 2. The methods will be refactored to be more modular and reusable. The `individuals_and_arrows` method will be the core of the new `parse()` method.

### Functions

-   **`_verify_is_ric_class`**: Verifies if a class is a known RiC-O class.
    -   **`AGENTS.md` Reflection:** This logic will be part of the `gbad_core/ric_ontology.py` module.
-   **Identifier Cleaning Functions (`_handle_spaces`, `_replace_metacharacter`, `_replace_metacharacters`)**: These are for cleaning up node labels to create valid IRIs.
    -   **`AGENTS.md` Reflection:** This logic will be preserved in the refactored `DrawIOParser` but will be made more configurable.
-   **Data Assembly Functions (`_add_individual_type`, `individual_blocks`)**: These assemble the parsed data.
    -   **`AGENTS.md` Reflection:** The new `DrawIOParser` will directly return a structured representation, so these functions will be replaced by the logic within the `parse()` method.
-   **Serialization Functions (`_infer_type`, `_serialise_facts`, `_serialise_block`, `_preamble`, `serialise`)**: These are for generating the OWL Manchester syntax output.
    -   **`AGENTS.md` Reflection:** This entire serialization logic will be removed from the `drawio` module. The `DrawIOParser` will only parse, not serialize. The serialization to RML will be handled by the `RMLGenerator` in `gbad_core/rml.py` (Task 3).
-   **Argument Parsing and Execution Functions (`_parse...`, `_arguments_parser`, `_run`, `_main`)**: These are for running the script from the command line.
    -   **`AGENTS.md` Reflection:** This will be replaced by a new CLI in `main.py` (Task 8), which will use the `gbad_core` library.

## `map_schema.py` Analysis

This script is the heart of the current RML generation process. It reads a TTL file (presumably generated from a Draw.io diagram), preprocesses a source CSV, and generates an RML mapping file.

### Global Variables

-   **`BASE_URI`**: Hardcoded base URI.
    -   **`AGENTS.md` Reflection:** To be moved to `gbad_core/config.py`.
-   **CSV Column Names and Labels (`SISN`, `DATEEX_COLS`, etc.)**: These are hardcoded and specific to the current datasets.
    -   **`AGENTS.md` Reflection:** This kind of configuration will be externalized. The `CSVPreprocessor` in `gbad_core/preprocessing.py` (Task 3) will be configured with these details, rather than having them hardcoded.
-   **Placeholder Masks (`rico_version_mask`, etc.)**: Used for string replacement.
    -   **`AGENTS.md` Reflection:** The new `RMLGenerator` in `gbad_core/rml.py` (Task 3) will have a more robust and configurable mechanism for handling such placeholders.
-   **`rico_authtp_dict`**: Hardcoded mapping logic.
    -   **`AGENTS.md` Reflection:** This is a prime example of hardcoded logic that will be made configurable, likely as part of the preprocessing or RML generation configuration.
-   **RML Generation Constants (`uuid_label`, etc.)**: These will be part of the `RMLGenerator`'s internal implementation.

### Functions

-   **`add_suppl_triples`**: Adds triples from supplementary files.
    -   **`AGENTS.md` Reflection:** This functionality will be part of the `GraphMerger` in `gbad_core/graph_merger.py` (Task 5), which will be able to merge any number of RDF files.
-   **`add_preprocess`**, **`auth_preprocess`**: These functions contain highly specific preprocessing logic for the `add` and `auth` schemas.
    -   **`AGENTS.md` Reflection:** As per Task 3, this logic will be moved into a new `CSVPreprocessor` class in `gbad_core/preprocessing.py`. The steps will be made generic and configurable, so they can be applied to any CSV file, not just the current ones.
-   **`__init__` (main function)**: This function is doing too many things. It's acting as the main orchestrator for the entire RML generation process.
    -   **`AGENTS.md` Reflection:** The responsibilities of this function will be broken down and distributed among the new classes as per the refactoring plan:
        -   **Orchestration:** The new `Pipeline` class in `gbad_core/pipeline.py` (Task 6) will be responsible for orchestrating the overall process.
        -   **RML Generation:** The `RMLGenerator` class in `gbad_core/rml.py` (Task 3) will handle the core RML generation logic.
        -   **Configuration:** All hardcoded paths, URIs, and namespaces will be managed by the `gbad_core/config.py` module (Task 7).
-   **`disaggregate_rico_authtp`**, **`disaggregate_refd_file`**: These functions are responsible for the complex and dataset-specific placeholder replacement logic.
    -   **`AGENTS.md` Reflection:** This logic will be generalized and made part of the `RMLGenerator`'s configurable placeholder handling mechanism.
-   **FnO Functions (`fno_map_value_unless_isnull`, `fno_map_this_string_match`)**: These are for creating conditional RML mappings.
    -   **`AGENTS.md` Reflection:** These are useful patterns and will be preserved as helper functions within the `gbad_core/rml.py` module, to be used by the `RMLGenerator`.
-   **Argument Parsing**: The command-line argument parsing will be replaced by the new CLI in `main.py` (Task 8).

## `map_rml.py` Analysis

This script is responsible for executing the RML mapping using an external JAR file and then post-processing the resulting RDF graph.

### Global Variables

-   **`BASE_URI`**: Hardcoded base URI.
    -   **`AGENTS.md` Reflection:** To be moved to `gbad_core/config.py`.

### Functions

-   **`map_rml`**: The main function for orchestrating the RML mapping.
    -   **`AGENTS.md` Reflection:** The logic of this function will be moved into the new `RMLMapper` class in `gbad_core/mapper.py` (Task 4).
        -   The hardcoded paths will be replaced by configuration values.
        -   The execution of the RML mapper will be handled by a method in the `RMLMapper` class. This will abstract away the details of whether it's a JAR file or a Python library.
-   **`postprocess`**: This function contains the data cleanup logic.
    -   **`AGENTS.md` Reflection:** As per Task 4, this logic will be moved to a new `gbad_core/postprocessing.py` module. The cleanup steps (e.g., `remove_false_agentcontrolrelation`) will be made configurable, so they can be selectively applied.
-   **`add_suppl_triples`**: Adds triples from supplementary files.
    -   **`AGENTS.md` Reflection:** This is redundant with the same function in `map_schema.py`. This functionality will be handled by the `GraphMerger` in `gbad_core/graph_merger.py` (Task 5).
-   **Argument Parsing**: The command-line argument parsing will be replaced by the new CLI in `main.py` (Task 8).

## `merge_graphs.py` Analysis

This script is responsible for merging multiple RDF files into a single N-Quads file.

### Global Variables

-   **`BASE_URI`**: Hardcoded base URI.
    -   **`AGENTS.md` Reflection:** To be moved to `gbad_core/config.py`.

### Functions

-   **`consolidate_rdf_files`**: The main function for merging graphs.
    -   **`AGENTS.md` Reflection:** The logic of this function will be moved into the new `GraphMerger` class in `gbad_core/graph_merger.py` (Task 5).
        -   The `GraphMerger` class will have a `merge` method that takes a list of file paths or `rdflib.Graph` objects.
        -   The named graph generation logic using SHA256 hashes is a good practice and will be preserved in the new `GraphMerger` class.
-   **Argument Parsing**: The command-line argument parsing will be replaced by the new CLI in `main.py` (Task 8).
