# Guidance for Human Staff

This document outlines the specific action items and decision points where your input is required to support the agentic staff in the GBAD refactoring project. Your domain expertise is crucial for the success of this project.

## 1. How to Collaborate

-   **Review and Respond to Agent Questions:** The agentic staff will use the `request_user_input` tool to ask for clarification on the items listed below. Please provide timely and specific answers to these questions.
-   **Provide Concrete Examples:** When possible, please provide concrete examples to illustrate your answers. This will help the agentic staff to understand the requirements and implement them correctly.

## 2. Specific Action Items and Decision Points

### `draw_io_parser.py`

-   **Action Item 1: Define Behavior for Ambiguous Node Structures.**
    -   **Context:** The current parser assumes a specific structure for individual nodes (a parent cell with an identifier and a child cell with a RiC-O class).
    -   **Question:** How should the system handle nodes that do not follow this structure?
    -   **Options:**
        -   a) Ignore the node and log a warning.
        -   b) Raise an error and stop the parsing process.
        -   c) Attempt to infer the structure based on a set of predefined rules.
    -   **Input Needed:** Please choose one of the options above and provide any additional rules or heuristics that should be applied.

-   **Action Item 2: Define Handling for Non-RiC-O Properties.**
    -   **Context:** The current parser can be configured to allow arrows with labels that are not known RiC-O properties.
    -   **Question:** How should these non-standard properties be handled in the refactored pipeline?
    -   **Options:**
        -   a) Map them to a default property (e.g., `rdfs:seeAlso`).
        -   b) Make this a configurable option in the pipeline, allowing the user to specify the desired behavior.
        -   c) Treat them as errors.
    -   **Input Needed:** Please choose one of the options above.

### `map_schema.py`

-   **Action Item 3: Provide Rationale for Preprocessing Logic.**
    -   **Context:** The `add_preprocess` and `auth_preprocess` functions contain complex, hardcoded logic for transforming CSV columns.
    -   **Question:** What is the business rationale behind these transformations? For example, why is the `INDEXPROV` column split by adjacent case?
    -   **Input Needed:** Please provide a detailed explanation of the business rules and data semantics for the preprocessing steps. This will enable the agentic staff to create a more robust and configurable preprocessing module.

-   **Action Item 4: Define Requirements for Disaggregation Logic.**
    -   **Context:** The `disaggregate_rico_authtp` and `disaggregate_refd_file` functions are highly specific to the current dataset.
    -   **Question:** To generalize this, what are the underlying requirements? Is the goal to create different types of entities based on the `AUTHTP` value, or to handle different reference schemes in the `REFD_FILE`?
    -   **Input Needed:** Please provide a clear definition of the different entity types and reference schemes that the system needs to support.

-   **Action Item 5: Identify Additional Conditional Mapping Scenarios.**
    -   **Context:** The script uses FnO to create conditional mappings.
    -   **Question:** Are there other scenarios where conditional logic is needed? For example, should we be able to define rules like "if column A has value X, then create property Y"?
    -   **Input Needed:** Please provide a list of any additional conditional mapping scenarios that the refactored system should support.

### `map_rml.py`

-   **Action Item 6: Provide Rationale for Post-processing Rules.**
    -   **Context:** The `postprocess` function contains several hardcoded cleanup rules.
    -   **Question:** What are the data quality issues that these rules are trying to address? Are there other known issues that should be handled?
    -   **Input Needed:** Please provide a list of all known data quality issues and the desired resolution for each.

### `merge_graphs.py`

-   **Action Item 7: Define Provenance Requirements.**
    -   **Context:** The script uses a SHA256 hash of the file content to generate a named graph URI for provenance.
    -   **Question:** Is this sufficient for all use cases? Would it be useful to also include other metadata in the named graph, such as the file name, creation date, or the user who uploaded it?
    -   **Input Needed:** Please provide a list of all metadata that should be included in the named graph for provenance.
