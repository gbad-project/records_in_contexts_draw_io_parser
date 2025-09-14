# Implementation Plan for TypeScript Client-Side Refactor

## Goal
Port the existing four Python scripts (`draw_io_parser.py`, `map_schema.py`, `map_rml.py`, `merge_graphs.py`) to a fully client-side TypeScript + React stack. The new pipeline will parse Draw.io diagrams and companion CSV files directly into RDF triples, apply two rounds of post-processing, and merge results into a single quadstore with each conversion placed in its own named graph.

## Conventions for All AI Agents
- **Comments**: use only these forms inside source files
  - `AICODE-TODO:` a task for another agent to implement later
  - `AICODE-NOTE:` explanatory note, no action required
  - `AICODE-ASK:` question that requires human feedback
- **Testing scripts**: every task that adds or modifies code must provide a bash script in `scripts/` that runs the relevant tests and writes a log at `logs/<agent>-log-<timestamp>.log` containing start and end timestamps.
- **Reports**: each agent writes a report to `reports/<agent>-report-<timestamp>.md` summarizing work and any follow‑ups.
- **Report mapping**: [reports/codex-report-20250913201852.md](reports/codex-report-20250913201852.md) links report files to their related tasks.
- **Globally relevant reports**:
  - [reports/jules-report-20250913113633.md](reports/jules-report-20250913113633.md)
  - [reports/architect-report-20250913180723.md](reports/architect-report-20250913180723.md)
  - Chief architect review: [reports/codex-report-20250914184816.md](reports/codex-report-20250914184816.md)
- **Commit discipline**: one focused task per commit, tests must run before committing, logs and reports included in the commit.

## Proposed Project Structure
```
/AGENTS.md                 – this plan
/logs/                     – test logs
/reports/                  – agent reports
/scripts/                  – bash test runners
/src/
  lib/                     – shared utilities (RDF helpers, CSV helpers)
  drawio/                  – Draw.io parser
  csv/                     – CSV preprocessing
  mapping/                 – CSV → RDF mapping engine
  postprocess/             – graph post-processing steps
  merge/                   – final merge and second-pass processing
  ui/                      – React components
/tests/                    – Vitest unit/integration tests
```

## High-Level Phases & Tasks
Each task references the directory where work occurs and the test script to be added.

### Phase 0 – Toolchain & Skeleton
1. **P0T1 – Node toolchain setup**
   - Add `package.json` with Volta pin (Node 20), Bun as runtime, scripts (`dev`, `build`, `test`).
   - Add `tsconfig.json`, `.eslintrc`, `.prettierrc` as needed.
   - Test script: `scripts/test-toolchain.sh` runs `bun test` (should exit 0 even when no tests).
   - Details about how this task description was created are available from report file(s) located at [reports/architect-report-20250913180723.md](reports/architect-report-20250913180723.md).
2. **P0T2 – Directory scaffolding**
   - Create the `src` subfolders listed above with placeholder index files.
   - Add `tests/setup.ts` if needed for Vitest.
   - Test script: reuse `scripts/test-toolchain.sh`.

3. **AICODE-TODO: P0T3 – Python/Conda environment for regression tests**
   - Provision the `gbad-next` Conda environment using `next/environment.yml` and `conda env create -n gbad-next -f next/environment.yml`.
   - Ensure the environment is activated with `conda activate gbad-next` so legacy Python scripts and OpenJDK dependencies are available for regression testing.
   - Details about how this task description was created are available from report file located at [reports/codex-report-20250914021652.md](reports/codex-report-20250914021652.md).
   - **AICODE-NOTE:** Provide `scripts/test-conda-env.sh` to verify the environment is created and activated, logging to `logs/test-conda-env-<timestamp>.log`.

### Phase 1 – RDF & CSV Utility Layer
1. **P1T1 – RDF helpers** (`src/lib/rdf.ts`) <!-- reviewed -->
   - **Goal**: Create a robust, well-tested wrapper around the `n3` library to provide a consistent and easy-to-use API for creating and serializing RDF throughout the application.
   - **Details about how this task description was created are available from report file located at [reports/jules-report-1757793520.md](reports/jules-report-1757793520.md).**

   - **AICODE-TODO: P1T1.1 - Add `n3` dependency.**
     - The first step is to add the `n3` library to the project's dependencies. Run `bun add n3 @types/n3`.
   - **AICODE-TODO: P1T1.2 - Create the module and re-export core types.**
     - In `src/lib/rdf.ts`, import and re-export the following core types from the `n3` library for easy access in other parts of the codebase: `NamedNode`, `Literal`, `BlankNode`, `Quad`, `Store`, `DataFactory`.
   - **AICODE-TODO: P1T1.3 - Implement Term Factory Functions.**
     - Create and export a set of thin wrapper functions around `n3.DataFactory` to simplify the creation of RDF terms.
     - `export const namedNode = DataFactory.namedNode;`
     - `export const literal = DataFactory.literal;`
     - `export const blankNode = DataFactory.blankNode;`
     - `export const quad = DataFactory.quad;`
   - **AICODE-TODO: P1T1.4 - Implement Store Creation Function.**
     - Create and export a function `createStore(quads?: Quad[]): Store`. This function will instantiate a new `n3.Store`, optionally seeding it with an initial array of quads.
   - **AICODE-TODO: P1T1.5 - Implement Serialization Function.**
     - Create and export an `async` function `serialize(store: Store, format: 'Turtle' | 'N-Quads' | 'N-Triples'): Promise<string>`. This function will use an `n3.Writer` to serialize the store's contents into a string, returning a `Promise` that resolves with the result.
   - **AICODE-TODO: P1T1.6 - Set up the Test File.**
     - Create `tests/lib/rdf.test.ts`. Import the necessary functions from `src/lib/rdf.ts` and testing utilities from `vitest`.
   - **AICODE-TODO: P1T1.7 - Test Term Creation Functions.**
     - Write individual tests for `namedNode`, `literal`, and `blankNode` to verify that they return objects with the expected `termType` and `value`.
   - **AICODE-TODO: P1T1.8 - Test `createStore` Function.**
     - Test `createStore` with no arguments (to ensure it returns an empty store) and with an array of quads (to ensure it returns a store with the correct size).
   - **AICODE-TODO: P1T1.9 - Test `serialize` Function.**
     - Test `serialize` for `'Turtle'`, `'N-Quads'`, and `'N-Triples'` formats. Verify the output is a valid string. Test with an empty store.
   - **AICODE-TODO: P1T1.10 - Test Error Handling.**
     - Test that the `serialize` function throws an error when an invalid format string is provided.
   - **Test script**: `scripts/test-lib-rdf.sh` → `bun test tests/lib/rdf.test.ts`.
2. **P1T2 – CSV helpers** (`src/lib/csv.ts`)
   - Utilities for loading CSV in browser and applying transformations.
   - Implement async `loadCsv` returning array of records.
   - Unit tests: `tests/lib/csv.test.ts`.
   - Test script: `scripts/test-lib-csv.sh`.
   - Details about how this task description was created are available from report file located at [reports/codex-report-20250913222101.md](reports/codex-report-20250913222101.md).
   - **AICODE-TODO: P1T2.1 - Core CSV Loader Implementation**
       - Implement `async function loadCsv(input: File | string, opts?: { delimiter?: string; }): Promise<Record<string, string>[]>`.
       - Use a browser-friendly parser (e.g., `PapaParse`) to read the CSV text.
       - Treat the first row as headers, ensure all cell values remain strings, strip UTF-8 BOM, and skip empty lines.
   - **AICODE-TODO: P1T2.2 - Transformation Pipeline Support**
       - Define `type CsvTransform = (row: Record<string, string>) => Record<string, string>`.
       - Apply transforms sequentially to each parsed row in an immutable way.
       - Provide a helper `applyTransforms(records: Record<string, string>[], transforms: CsvTransform[]): Record<string, string>[]`.
   - **AICODE-TODO: P1T2.3 - Error Handling and Validation**
       - Detect malformed CSV (e.g., inconsistent column counts) and surface descriptive errors.
       - Handle missing headers by throwing or returning a structured error object.
   - **AICODE-TODO: P1T2.4 - Testing Scaffold**
       - `tests/lib/csv.test.ts` should verify basic parsing, application of a transform (e.g., trimming whitespace), handling of BOMs, and skipping empty lines.
       - `scripts/test-lib-csv.sh` should run `bun test tests/lib/csv.test.ts` and log results to `logs/test-lib-csv-<timestamp>.log`.
   - **Relevant Files:**
       - `gbad/converter/preprocessors.py` – demonstrates loading CSV data as strings and column transformation patterns.
       - `map_schema.py` – uses `SourceCSVPreprocessor` and `column_split` to showcase transformation needs.
3. **P1T3 – Prefix Expansion Utility** (`src/lib/prefix.ts`)
    - **Goal**: Create a simple, configurable utility for expanding CURIEs (e.g., `rico:RecordSet`) into full IRIs.
    - **Details about how this task description was created are available from report file located at [reports/jules-report-20250913130525.md](reports/jules-report-20250913130525.md).**
    - **AICODE-TODO: P1T3.1 - Implement Prefix Map and Expansion Function.**
        - Create a function `createPrefixExpander(prefixMap: Record<string, string>)`. It takes a dictionary mapping prefixes to their IRI bases (e.g., `{ rico: "https://www.ica.org/standards/RiC/ontology#" }`).
        - The function should return another function, `expand(curie: string): string`.
        - The `expand` function should take a CURIE string, split it into prefix and local name, look up the prefix in the map, and return the concatenated full IRI.
        - If the prefix is not found or the input is not a valid CURIE, it should handle the error gracefully (e.g., return the original string or throw a configured error).
        - **AICODE-NOTE:** Decide and document a single strategy for unknown prefixes so downstream modules behave consistently.
    - **AICODE-NOTE: Usage**: This utility will not be used by the Draw.io parser (P2T1) itself, but will be used by later stages in the pipeline (like the mapping engine in P4) after the raw CURIEs have been extracted.
    - **Tests**: `tests/lib/prefix.test.ts`. Test the expander with valid CURIEs, unknown prefixes, and malformed inputs.
    - **Test script**: `scripts/test-lib-prefix.sh`.

### Phase 2 – Draw.io Parsing
1. **P2T1 – Ontology-Agnostic XML Parser** (`src/drawio/parser.ts`) <!-- reviewed -->
   - **Goal**: To parse the structural and semantic information from a Draw.io XML file into a generic, ontology-agnostic intermediate representation. This module will **not** validate identifiers against any specific ontology. Its sole job is to faithfully translate the diagram's structure.
   - Details about how this task description was created are available from report file(s) located at [reports/jules-report-20250913130525.md](reports/jules-report-20250913130525.md).

   - **AICODE-TODO: P2T1.1 - Define Generic Data Structures** (`src/drawio/model.ts`)
        - `DiagramNode`: Represents a swimlane or rounded rectangle.
            - `id`: `string`
            - `value`: `string` (The raw text content, e.g., `rr:template "/KB/{...}"` or `rico:RecordSet`)
            - `geometry`: `{ x: number; y: number; width: number; height: number; }`
            - `children`: `Map<string, DiagramNode>` (For nested nodes, like a class inside a swimlane)
        - `DiagramArrow`: Represents an arrow.
            - `id`: `string`
            - `label`: `string` (The raw text from the edge label, e.g., `rico:hasRecordSetType`)
            - `sourceId`: `string | null`
            - `targetId`: `string | null`
        - `ParsedDiagram`:
            - `nodes`: `Map<string, DiagramNode>`
            - `arrows`: `Map<string, DiagramArrow>`

    - **AICODE-TODO: P2T1.2 - Implement Core XML Parsing and Structuring** (`src/drawio/parser.ts`)
        - Create the main function `parseDrawio(xml: string): ParsedDiagram`.
        - Use an XML parser to convert the XML string into a JavaScript object.
        - Iterate through all `<mxCell>` elements and build a hierarchical tree of `DiagramNode` and `DiagramArrow` objects based on their `id` and `parent` attributes. This will create a structured representation of the raw graph.
        - Use the `extractTextFromHtml` utility (from P2T1.3) to clean the `value` of each cell.

    - **AICODE-TODO: P2T1.3 - Implement HTML Value Extraction** (`src/drawio/html-parser.ts`)
        - Create a utility function `extractTextFromHtml(html: string): string` to correctly parse the HTML embedded in `value` attributes.

    - **AICODE-TODO: P2T1.4 - Implement Arrow Source/Target Resolution** (`src/drawio/arrow-resolver.ts`)
        - Create a function `resolveArrowConnections(diagram: ParsedDiagram, config: { maxGap: number })`.
        - It will iterate through the arrows in the `ParsedDiagram` and resolve any missing `sourceId` or `targetId` by performing the geometric search against the `DiagramNode` geometries.

    - **AICODE-NOTE: Decoupling Principle**
        - This parser should not contain any hardcoded lists of classes or properties.
        - It should not perform any validation of whether a `value` like `"rico:RecordSet"` is a "known" class. It simply extracts the string. The responsibility for interpreting and validating these strings lies with downstream modules.

    - **AICODE-NOTE: Testing Strategy**
        - Create `tests/drawio/parser.test.ts`.
        - Unit test each helper function (`html-parser`, `arrow-resolver`).
        - For the main `parseDrawio` function, use the sample `.drawio` files as input and compare the output `ParsedDiagram` object against a stored JSON snapshot. This ensures the structural parsing is correct without depending on any specific ontology.
   - **Relevant Files**:
     - **Python Scripts**: `draw_io_parser.py`
     - **Test Scripts**: `tests/test_draw_io_parser.py`
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`, `tests/scripts/linux/test_run.sh`
     - **Referenced Files**:
       - `gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio`
       - `gbad/schema/authority/General Authority to RiC-O Model_2025-06-25_PZ.drawio`

### Phase 3 – CSV Preprocessing
1. **P3T1 – Custom preprocessing functions** (`src/csv/preprocess.ts`) <!-- reviewed -->
   - **Goal**: Port the CSV preprocessing logic from `map_schema.py` into a robust, well-tested TypeScript module. This module will accept raw CSV data as an array of objects and return a new array of objects with added and transformed columns. The logic is divided into three distinct pipelines based on a `schema_code`: `add`, `auth`, and `generic`.
   - Details about how this task description was created are available from report file(s) located at [reports/jules-report-20250913113633.md](reports/jules-report-20250913113633.md) and [reports/jules-report-20250913120003.md](reports/jules-report-20250913120003.md).

   - **AICODE-TODO: P3T1.1 - Implement a CSV Preprocessing Utility.**
     - Before implementing the specific pipelines, create a utility class or a set of functions in TypeScript that replicates the functionality of the `SourceCSVPreprocessor` class in `gbad/converter/preprocessors.py`.
     - This utility must provide methods for:
       - Loading and managing an array of CSV record objects.
       - Adding a new column to every record from a provided series/array of values.
       - Updating records from an external data source.
       - A generic `columnSplit` function that takes a source column name, a list of new column names, and a transformation function. This function should apply the transformation to the source column for each record and populate the new columns.

   - **AICODE-TODO: P3T1.2 - Implement the `add` Schema Preprocessing Pipeline.**
     - This pipeline transforms data for "Descriptions and Listings".
     - **Column Splits**: Implement the following transformations, using the utility from P3T1.1:
       1.  `FINDAID:FINDAIDLINK:FINDAID_URL` -> `FINDAID`, `FINDAIDLINK`, `FINDAID_URL`. Logic: split by ` : `.
       2.  `IIL:IIL_URL` -> `IIL`, `IIL_URL`. Logic: split by ` : `.
       3.  `INDEXPROV` -> `INDEXPROV_1` to `INDEXPROV_30`. Logic: split by adjacent-case (e.g., `FirstSecond` becomes `["First", "Second"]`).
       4.  `INDEXNAME` -> `INDEXNAME_1` to `INDEXNAME_30`. Logic: split by adjacent-case.
       5.  `INDEXSUB` -> `INDEXSUB_1` to `INDEXSUB_30`. Logic: split by adjacent-case.
       6.  `DATEOFF:OFFICEAB:AB_REFA:OFFICEC:C_REFA` -> 100 new columns (`DATEOFF_1`, `OFFICEAB_1`...`C_REFA_20`). Logic: split by ` : `.
       7.  For each `DATEOFF_{i}` (from 1 to 20) -> `DATEOFF_{i}_BEGINNING`, `DATEOFF_{i}_END`. Logic: split by `-`.
     - **Conditional Logic**:
       1.  For each record `i` from 1 to 20:
       2.  Create `ABC_REFA_{i}` by combining `C_REFA_{i}` and `AB_REFA_{i}` (preferring the value from `C_REFA_{i}`).
       3.  Create `OFFICE_TYPE_{i}` by taking the first character of `ABC_REFA_{i}`.
       4.  Create `OFFICEABC_{i}` by selecting the value from `OFFICEAB_{i}` if `OFFICE_TYPE_{i}` is 'A' or 'B', or from `OFFICEC_{i}` if it's 'C'.

   - **AICODE-TODO: P3T1.3 - Implement the `auth` Schema Preprocessing Pipeline.**
     - This pipeline transforms data for "Authorities".
     - **Regex-based Categorization**:
       1.  For `AUTHTP_1` and `AUTHTP_2`, apply a categorization logic based on regex matching.
       2.  For each input column, generate a set of new columns: `RICO_AUTHTP_NEW_{i}`, `RICO_AUTHTP_LABEL_{i}`, and specific type columns (`..._CORPORATEBODY_{i}`, `..._FAMILY_{i}`, etc.).
       3.  The mapping from regex to category is defined in the `rico_authtp_dict` in `map_schema.py`. This dictionary must be replicated in TypeScript.
     - **External Data Merge**:
       1.  Implement logic to read a secondary CSV file (`gbad/mapping/source/New-export-of-Government-authorities-with-correct-Dates-of-Existence-xlsx.csv`).
       2.  Use the data from this file to update the `DATEEX_BEGINNING` and `DATEEX_END` columns in the primary dataset, matching on the `SISN` index.

   - **AICODE-NOTE: Data Contract and Models.**
     - The input data for the `add` pipeline will be records with columns like `SISN`, `FINDAID:FINDAIDLINK:FINDAID_URL`, `INDEXPROV`, etc.
     - The output for the `add` pipeline will be records containing all original columns plus ~150 new columns (e.g., `FINDAID`, `FINDAIDLINK`, `INDEXPROV_1`, `DATEOFF_1_BEGINNING`, `OFFICEABC_1`, etc.).
     - The input data for the `auth` pipeline will have columns like `SISN`, `AUTHTP_1`, `AUTHTP_2`.
     - The output for the `auth` pipeline will have all original columns plus 12 new `RICO_AUTHTP_*` columns.

   - **AICODE-NOTE: Testing Strategy.**
     - The existing Python unit test (`tests/test_map_schema_generic.py`) is insufficient as it does not test the `add` or `auth` preprocessing logic.
     - A new, comprehensive test suite **must** be created in `tests/csv/preprocess.test.ts`.
     - Use the test files `gbad/mapping/source/tests/test_description_tailshuf_100.csv` and `gbad/mapping/source/tests/test_authority_tailshuf_100.csv` as the basis for your test cases.
     - Create specific unit tests for each transformation:
       - Test the adjacent-case split logic.
       - Test the colon-splitting logic with both full and partial data.
       - Test the `OFFICEABC` conditional logic.
       - Test each regex category in the `AUTHTP` mapping.
       - Mock the external date file to test the data merging logic.

   - **AICODE-NOTE: Discovered Issues.**
     - The paths to test data in the integration scripts (e.g., `tests/scripts/linux/test_add.sh`) are incorrect. They point to `tests/test_...` when the files are actually in `gbad/mapping/source/tests/`. Be aware of this if you need to run those scripts.

   - **Relevant Files**:
     - **Source Logic**: `map_schema.py`, `gbad/converter/preprocessors.py`
     - **Test Case Definition (Input)**: `gbad/mapping/source/tests/test_description_tailshuf_100.csv`, `gbad/mapping/source/tests/test_authority_tailshuf_100.csv`
     - **Test Case Definition (Output)**: The corresponding files in `gbad/mapping/source/preprocessed/tests/`
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`

### Phase 4 – Mapping Engine (RML removal)
1. **P4T1 – Mapping model** (`src/mapping/model.ts`) <!-- reviewed -->
   - **Goal**: Define the core TypeScript interfaces that represent the mapping instructions. This model will be generated from the `ParsedDiagram` (from `P2T1`) and used by the mapping engine (`P4T2`) to convert CSV data into RDF triples. This replaces the need for an intermediate RML file.
   - **Details about how this task description was created are available from report file located at [reports/jules-report-20250913142338.md](reports/jules-report-20250913142338.md).**

   - **AICODE-TODO: P4T1.1 - Define the `ObjectMap` Interfaces.**
        - Create a base interface `ObjectMap` with a `type` property.
        - Create specific interfaces that extend `ObjectMap` for each type of object generation:
            - `ReferenceObjectMap`: `{ type: 'reference'; column: string; }`
            - `TemplateObjectMap`: `{ type: 'template'; template: string; }`
            - `ConstantObjectMap`: `{ type: 'constant'; value: string; }`
            - `ParentTriplesMapObjectMap`: `{ type: 'parentTriplesMap'; parentTriplesMapId: string; joinCondition?: { child: string; parent: string; }; }`

   - **AICODE-TODO: P4T1.2 - Define the `PredicateObjectMap` Interface.**
        - Create an interface `PredicateObjectMap`.
        - It should have a `predicate` property (`string`) to hold the predicate URI.
        - It should have an `objectMaps` property, which is an array of the `ObjectMap` union type (`(ReferenceObjectMap | TemplateObjectMap | ConstantObjectMap | ParentTriplesMapObjectMap)[]`).

   - **AICODE-TODO: P4T1.3 - Define the `TriplesMap` Interface.**
        - Create the main `TriplesMap` interface.
        - It needs the following properties:
            - `id`: `string` (A unique identifier for the map).
            - `rdfClass`: `string` (The `rdf:type` of the resource to be created).
            - `subjectTemplate`: `string` (The template for the subject URI).
            - `logicalSource`: `{ path: string; }` (Information about the source data).
            - `condition`: `{ column: string; value: 'not null'; }` (Optional, for conditional generation).
            - `predicateObjectMaps`: `PredicateObjectMap[]` (An array of predicate-object maps).

   - **AICODE-TODO: P4T1.4 - Define the main `MappingModel` and Helper Function.**
        - Define a type alias `MappingModel` as `TriplesMap[]`.
        - Define the signature for the helper function that will create this model: `buildMappingModel(parsedDiagram: ParsedDiagram): MappingModel`. The implementation of this function is not part of this task, but its signature should be defined for clarity.

   - **AICODE-NOTE: Data Contract.**
        - The `MappingModel` is the complete, declarative definition of the CSV-to-RDF mapping.
        - The `buildMappingModel` function will be responsible for converting the generic `ParsedDiagram` into this specific `MappingModel`. This is where the logic from `map_schema.py`'s `uriref_str_to_map` will be adapted.
        - The `mapCsvToRdf` function (`P4T2`) will take this `MappingModel` as input and execute the mapping.
   - **Relevant Files**:
     - **Python Scripts**: `map_schema.py`, `map_rml.py`
     - **Test Scripts**: `tests/mapping/model.test.ts`
     - **Test script**: `scripts/test-mapping-model.sh`
2. **P4T2 – CSV → RDF triple conversion** (`src/mapping/mapper.ts`) <!-- reviewed -->
   - Implement function `mapCsvToRdf(mapping: MappingModel, csv: Record[]): Dataset`.
   - Integrate preprocessing from Phase 3.
   - First round post‑processing hooks (`applyInitialPostprocess(dataset)` in `src/postprocess/initial.ts`).
   - Details about how this task description was created are available from report file(s) located at [reports/jules-report-20250913121644.md](reports/jules-report-20250913121644.md).

- **AICODE-TODO: P4T2.1 - Implement Basic Mapping Loop and Subject Generation** (`src/mapping/mapper.ts`)
   - **Goal**: Create the `mapCsvToRdf` function skeleton and implement the core iteration logic. This first step will focus on generating the primary resources (subjects) and their `rdf:type` declarations.
   - **Implementation**:
     - Create the `mapCsvToRdf` function.
     - Implement the main loops for iterating over `csvData` records and `mapping.TriplesMap` definitions.
     - For each record and triples map, generate the subject `NamedNode` by substituting placeholders in the `subjectTemplate` with values from the record.
     - Add the `rdf:type` triples for the subject using the `triplesMap.rdfTypes` array.
   - **Tests**: `tests/mapping/mapper.test.ts` should verify that subjects and their types are created correctly for a basic mapping.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

- **AICODE-TODO: P4T2.2 - Add Object Generation for Literals and Constants** (`src/mapping/mapper.ts`)
   - **Goal**: Extend the mapper to handle the simplest object types: literals from CSV columns and constant values.
   - **Implementation**: In the `predicateObjectMaps` loop, add logic to handle object maps of type `reference` and `constant`.
     - For `reference`, create an `rdf:Literal` using the value from the specified CSV column in the current record.
     - For `constant`, create an `rdf:Literal` or `rdf:NamedNode` from the hardcoded value.
   - **Tests**: Extend `tests/mapping/mapper.test.ts` to test that properties with literal and constant objects are correctly generated.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

- **AICODE-TODO: P4T2.3 - Implement Conditional Triple Generation** (`src/mapping/mapper.ts`)
   - **Goal**: Implement the "not null" condition to prevent creating triples when source data is missing.
   - **Implementation**: Wrap the predicate-object generation logic in a conditional check. If any source column for an object is empty, skip creating that triple.
   - **Tests**: `tests/mapping/mapper.test.ts` must include cases where source data is missing and verify that triples are *not* generated.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

- **AICODE-TODO: P4T2.4 - Add Object Generation for Joins** (`src/mapping/mapper.ts`)
   - **Goal**: Implement support for linking between resources using `parentTriplesMap`.
   - **Implementation**: Add logic to handle `objectMap` of type `parentTriplesMap`, which requires looking up the target `TriplesMap` and generating its subject IRI using the same current CSV record.
   - **Tests**: `tests/mapping/mapper.test.ts` needs a test case with a mapping that uses `parentTriplesMap`.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

- **AICODE-TODO: P4T2.5 - Implement Initial Post-processing** (`src/postprocess/initial.ts`)
   - **Goal**: Implement the `remove_shorter_duplicate_labels` algorithm and integrate it into the pipeline.
   - **Implementation**: Create `applyInitialPostprocess` in `src/postprocess/initial.ts` with the required logic. Call it from `mapCsvToRdf` after the main mapping loop.
   - **Tests**: `tests/postprocess/initial.test.ts` should be created to specifically test the algorithm.
   - **Test Script**: `scripts/test-postprocess-initial.sh`

- **Relevant Files**:
     - **Python Scripts**: `map_rml.py`
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`, `tests/scripts/linux/test_rg_1-429.sh`, `tests/scripts/linux/test_run.sh`
     - **Referenced Files**:
       - RML files generated by `map_schema.py` (e.g., `gbad/schema/generic/generic/generic_schema.rml`)
       - `gbad/mapping/target/`

### Phase 5 – Postprocessing Modules
1. **P5T1 – Initial postprocessing** (`src/postprocess/initial.ts`) <!-- reviewed -->
   - **Goal**: Port the RDF graph post-processing logic from the Python script `map_rml.py` into a TypeScript module. This module will provide a function `applyInitialPostprocess(dataset)` that takes an `n3.js` dataset and applies a series of transformations to clean up and enrich the data.
   - Details about how this task description was created are available from report file(s) located at [reports/jules-report-1757793156.md](reports/jules-report-1757793156.md).

   - **AICODE-TODO: P5T1.1 - Implement the `remove_shorter_duplicate_labels` Algorithm.**
     - **Description**: This is the primary active post-processing step. It identifies `rico:RecordSet` resources that have exactly two `rdfs:label` predicates. It then checks if both labels start with the text of the record set's `add:CurrentReferenceCode`. If this condition is met, the algorithm must remove the triple containing the shorter of the two labels.
     - **Implementation Details**:
       - The function should accept an `n3.js` `Dataset` as input.
       - It needs to query the dataset to find all `?subject a rico:RecordSet`.
       - For each subject, get all its `rdfs:label` values.
       - If there are two labels, find the linked identifier via `rico:hasOrHadIdentifier` that is of type `add:CurrentReferenceCode`.
       - The reference code text must be extracted from the identifier's label (e.g., extract "C 1" from `"C 1 (Current Reference Code)"`).
       - Perform the string comparison and remove the triple with the shorter label from the dataset.

   - **AICODE-TODO: P5T1.2 - Implement Supplemental Triple Loading.**
     - **Description**: The Python script has a function `add_suppl_triples` that loads all `.ttl` files from the `gbad/schema` directory and merges them into the main graph. This functionality needs to be replicated.
     - **Implementation Details**:
       - Create a helper function that can fetch and parse multiple Turtle files from a given directory path.
       - The `applyInitialPostprocess` function should orchestrate this, adding the triples from `gbad/schema/*.ttl` to the dataset it is processing.
       - This will likely require an async function and fetching files over HTTP in a browser context.

   - **AICODE-TODO: P5T1.3 - Stub Out Inactive Post-processing Functions.**
     - **Description**: The Python script contains two commented-out functions: `remove_false_agentcontrolrelation` and `remove_false_authtp`. While they are not currently active, they should be ported as stubbed-out functions in the TypeScript module for future use.
     - **Implementation Details**:
       - Create empty functions `removeFalseAgentControlRelation(dataset)` and `removeFalseAuthTp(dataset)` in `src/postprocess/initial.ts`.
       - Add comments inside each function explaining its original purpose based on the Python source.

   - **AICODE-NOTE: Function Signature and Data Contract.**
     - The main function to be exported from `src/postprocess/initial.ts` should be:
       ```typescript
       import { Dataset } from 'n3';

       export async function applyInitialPostprocess(dataset: Dataset): Promise<Dataset> {
         // ... implementation ...
         return dataset;
       }
       ```
     - The function will directly mutate the dataset passed to it.

   - **AICODE-NOTE: Testing Strategy.**
     - A new test suite `tests/postprocess/initial.test.ts` must be created.
     - **For `remove_shorter_duplicate_labels`**:
       - Create a test case with a sample `rico:RecordSet`.
       - The record set should have two `rdfs:label`s and a linked `add:CurrentReferenceCode`.
       - Case 1: Both labels start with the ref code. Verify the shorter label is removed.
       - Case 2: Only one label starts with the ref code. Verify no labels are removed.
       - Case 3: The record set has only one label, or three labels. Verify no labels are removed.
     - **For Supplemental Triple Loading**:
       - Mock the file fetching mechanism.
       - Provide a sample `.ttl` file content in the mock.
       - Verify that the triples from the mocked file are present in the dataset after the function runs.
     - The test script `scripts/test-postprocess-initial.sh` will execute `bun test tests/postprocess/initial.test.ts`.

   - **Relevant Files**:
     - **Source Logic**: `map_rml.py` (the `postprocess` function and its callees)
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`
     - **Supplemental Data**: `gbad/schema/`
2. **P5T2 – Merge-stage postprocessing** (`src/postprocess/merge.ts`)
   - New second-round transformations to apply after graph merging.
   - Stub out with `AICODE-TODO` where rules are unspecified.
   - **AICODE-ASK:** This feature was never implemented in the legacy code base, so an AI agent assigned this task should flag it for human review before proceeding.
   - Tests: `tests/postprocess/merge.test.ts`.
   - Test script: `scripts/test-postprocess-merge.sh`.

### Phase 6 – Graph Merge & Named Graph Handling
1. **P6T1 – Merge utility** (`src/merge/merger.ts`) <!-- reviewed -->
   - **Goal**: Create a client-side utility that takes multiple RDF graph datasets (as strings or `n3.Store` objects) and merges them into a single N-Quads string. Each input graph will be placed into its own deterministically generated named graph.
   - Details about how this task description was created are available from report file(s) located at [reports/jules-report-20250913125101.md](reports/jules-report-20250913125101.md).
   - **AICODE-TODO: P6T1.1 - Implement the Named Graph URI Generation Logic.**
     - Create a function `generateNamedGraphUri(graphContent: string): NamedNode`.
     - This function must replicate the exact logic from `create_named_graph_uri_from_file` in `merge_graphs.py`:
         1.  Take the graph content as a string.
         2.  Encode it to UTF-8 bytes.
         3.  Calculate the SHA-256 hash of the bytes.
         4.  Encode the hash using URL-safe Base64, removing padding (`=`).
         5.  Construct the `ni:///sha-256;{hash}` URI.
         6.  Generate a Version 5 UUID using `uuid.NAMESPACE_URL` and the `ni` URI.
         7.  Return an `rdf:NamedNode` with the value `urn:uuid:{uuid}`.
     - You will need a SHA-256 library and a UUID v5 library (e.g., `crypto-js`, `uuid`).
   - **AICODE-TODO: P6T1.2 - Implement the Core Merge Function.**
     - Create the main function `mergeGraphs(graphs: { name: string, data: Store }[]): Store`.
     - This function should accept an array of objects, where each object contains the data of a graph (as an `n3.Store`) and a name or identifier.
     - It should initialize a new `n3.Store` for the merged output.
     - For each input graph:
         1.  Serialize the input `Store` to a canonical string format (like N-Triples) to be used for hashing.
         2.  Generate the named graph URI using the function from P6T1.1.
         3.  Iterate through the quads of the input store and add them to the merged store, but replace their graph component with the newly generated named graph URI.
   - **AICODE-TODO: P6T1.3 - Implement Second-Pass Post-processing.**
     - Create a function `applyMergePostprocess(dataset: Store): Store`.
     - This function will contain the logic for the second round of transformations, as mentioned in the `AGENTS.md` high-level description.
     - For now, this can be a placeholder function that simply returns the dataset, with an `AICODE-TODO` comment inside it to indicate that the specific rules need to be implemented in task `P5T2`.
   - **AICODE-TODO: P6T1.4 - Create the Final Orchestration Function.**
     - Create a function `createMergedNquads(graphs: { name: string, data: Store }[]): string`.
     - This function will orchestrate the process:
         1.  Call `mergeGraphs` to get the merged dataset.
         2.  Call `applyMergePostprocess` on the result.
         3.  Serialize the final dataset to an N-Quads string.
         4.  Return the string.
   - **AICODE-NOTE: Testing Strategy.**
     - Create a test file `tests/merge/merger.test.ts`.
     - Create a test case that mimics the `test_run.sh` script.
     - Have two sample input graphs (as strings or loaded from test files).
     - Generate the merged N-Quads string using your new utility.
     - To verify the output, you will need to manually calculate the expected named graph URIs for your sample inputs. You can use the provided Python script `merge_graphs.py` on your sample data to get the exact expected URIs.
     - The test should then check that the output string contains the correct triples within the correct, expected named graphs.
   - **Relevant Files**:
     - **Python Scripts**: `merge_graphs.py`
     - **Integration Tests**: `tests/scripts/linux/test_run.sh`
     - **Referenced Files**:
       - Output of `map_rml.py` (e.g., `.ttl` or `.nt` files in `gbad/mapping/target/`)
       - `tests/data/store.nq`

### Phase 7 – React User Interface
1. **P7T1 – File upload & pipeline orchestration** (`src/ui/App.tsx` + components) <!-- reviewed -->
   - **Goal**: To create the main user interface for the application, which allows a user to upload their Draw.io and CSV files, configure and execute the entire end-to-end conversion pipeline in the browser, monitor its progress, and download the final, merged RDF graph.
   - **Details about how this task description was created are available from report file located at [reports/jules-report-20250913135639.md](reports/jules-report-20250913135639.md).**

   - **AICODE-TODO: P7T1.1 - UI Component Scaffolding (`src/ui/components/`)**
     - Create the main application layout component (`App.tsx`).
     - Create stateless, presentational components for the key UI areas:
       - `Header.tsx`: For the application title.
       - `FileUpload.tsx`: A component with two upload zones, one for the Draw.io XML file and one for the CSV data file.
       - `PipelineControls.tsx`: A component containing a "Run Pipeline" button and other configuration options.
       - `StatusBar.tsx`: A component to display the current status of the pipeline.
       - `ResultsViewer.tsx`: A component to display a summary of the results and a download button.

   - **AICODE-TODO: P7T1.2 - Prefix Map Configuration (`src/ui/components/PrefixMapEditor.tsx`)**
     - Create a new component `PrefixMapEditor.tsx` that allows the user to provide a prefix map.
     - This component should contain a `textarea` where a user can paste a JSON object representing the `prefixMap` (e.g., `{ "rico": "https://www.ica.org/standards/RiC/ontology#" }`).
     - Include basic validation to ensure the input is valid JSON.
     - The `PipelineControls.tsx` component should include this new editor.

   - **AICODE-TODO: P7T1.3 - State Management Setup (`src/ui/App.tsx`)**
     - In the main `App.tsx` component, set up state management to track the application state, including:
       - The uploaded Draw.io file content (`string | null`).
       - The uploaded CSV file content (`string | null`).
       - The user-provided `prefixMap` string and parsed object (`Record<string, string>`).
       - The selected `schema_code` (`'add' | 'auth' | 'generic'`).
       - The current pipeline status (`'idle' | 'parsing' | 'mapping' | 'merging' | 'complete' | 'error'`).
       - The intermediate and final results.
       - The final merged N-Quads string (`string | null`).
       - Any error messages.

   - **AICODE-TODO: P7T1.4 - File Input and Initial Orchestration**
     - Implement the logic for file selection and reading file content into state.
     - When the "Run Pipeline" button is clicked, begin the orchestration:
       1. Validate that all required inputs (files, prefix map JSON) are present and valid.
       2. **Create the configured prefix expander**: Call `createPrefixExpander` from `P1T3` with the user's `prefixMap` to get a configured `expand` function.
       3. Set the status to "Parsing Draw.io...".
       4. Call the `parseDrawio` function from `P2T1`.
       5. Set the status to "Loading CSV...".
       6. Call the `loadCsv` helper from `P1T2`.

   - **AICODE-TODO: P7T1.5 - CSV Preprocessing and Mapping Orchestration**
     - Continuing the pipeline orchestration:
       1. Set the status to "Preprocessing CSV...".
       2. Call the appropriate preprocessing function from `P3T1`.
       3. Set the status to "Mapping CSV to RDF...".
       4. **Inject the dependency**: Call the `mapCsvToRdf` function from `P4T2`, passing in the `ParsedDiagram`, the preprocessed CSV data, **and the configured `expand` function created in the previous step.**
       5. Store the resulting RDF `Dataset` in the state.

   - **AICODE-TODO: P7T1.6 - Post-processing and Merging Orchestration**
     - Continuing the pipeline:
       1. Set the status to "Post-processing...".
       2. Call `applyInitialPostprocess` from `P5T1`.
       3. Set the status to "Merging graphs...".
       4. Call `createMergedNquads` from `P6T1`.
       5. Store the final N-Quads string in the state.
       6. Set the status to "Complete".

   - **AICODE-TODO: P7T1.7 - Results Display and Download**
     - In `ResultsViewer.tsx`, display a summary of the successful conversion.
     - Enable the "Download .nq file" button.
     - Implement the download logic using a `Blob` and a temporary `<a>` tag.

   - **AICODE-NOTE: Testing Strategy**
     - Use Vitest and React Testing Library for component-level tests (`tests/ui/*.test.tsx`).
     - Mock the pipeline modules (`P1T1` through `P6T1`) to test the UI's state transitions and orchestration logic without running the actual heavy processing.
     - Test that file uploads update the state correctly.
     - Test that clicking the "Run Pipeline" button calls the mocked functions in the correct sequence and updates the status bar appropriately.
     - Test that the download button is enabled only upon completion and that it triggers a download.

   - **Test script**: `scripts/test-ui-app.sh`
   - **AICODE-NOTE:** Surface errors at each pipeline step and allow rerunning the workflow without a page reload.

### Phase 8 – Integration & Example Workflow
1. **P8T1 – End-to-end example** <!-- reviewed -->
   - **Goal**: Provide a runnable demonstration of the full TypeScript pipeline from Draw.io and CSV inputs to a merged N-Quads output.
   - **Details about how this task description was created are available from report file located at [reports/codex-report-20250914014845.md](reports/codex-report-20250914014845.md).**

   - **AICODE-TODO: P8T1.1 - Prepare example inputs.**
     - Place a representative `.drawio` diagram in `examples/example.drawio` (e.g., reuse `examples/f_4711_orville_wood_fonds.drawio`).
     - Place a matching CSV in `examples/example.csv`. Sample test CSVs are available in `gbad/mapping/source/tests/` such as `test_description_tailshuf_100.csv` and `test_authority_tailshuf_100.csv`.
   - **AICODE-TODO: P8T1.2 - Implement example workflow.**
     - Add `scripts/run-example.sh`.
     - Script steps:
       1. `bunx tsx src/examples/runExample.ts examples/example.drawio examples/example.csv`.
       2. `src/examples/runExample.ts` orchestrates:
          - `parseDrawio` from `P2T1` to parse the Draw.io diagram.
          - `loadCsv` from `P1T2` to read the CSV.
          - preprocessing utilities from `P3T1`.
          - `mapCsvToRdf` from `P4T2`.
          - `applyInitialPostprocess` from `P5T1`.
          - `createMergedNquads` from `P6T1`.
       3. Write the merged output to `examples/output.nq`.
   - **AICODE-TODO: P8T1.3 - Regression comparison.**
     - After running `scripts/run-example.sh`, execute legacy regression tests to ensure parity with the Python pipeline:
       - `python tests/test_draw_io_parser.py`
       - `python tests/test_map_schema_generic.py`
       - `bash tests/scripts/linux/test_add.sh`
       - `bash tests/scripts/linux/test_auth.sh`
       - `bash tests/scripts/linux/test_rg_1-429.sh`
       - `bash tests/scripts/linux/test_run.sh`
       - **AICODE-NOTE:** These regression tests require the fully provisioned `gbad-next` Conda environment from `next/environment.yml` to be activated, providing OpenJDK and other pinned dependencies.
       - **AICODE-NOTE:** Several legacy scripts reference outdated test data paths; update them to `gbad/mapping/source/tests` before executing.
   - **AICODE-TODO: P8T1.4 - Test script.**
     - Create `scripts/test-example.sh` that:
       1. Runs `bun scripts/run-example.sh`.
       2. Confirms `examples/output.nq` exists.
       3. Logs output to `logs/test-example-<timestamp>.log`.
   - **Relevant Files:**
     - `tests/test_draw_io_parser.py`
     - `tests/test_map_schema_generic.py`
     - `tests/scripts/linux/test_add.sh`
     - `tests/scripts/linux/test_auth.sh`
     - `tests/scripts/linux/test_rg_1-429.sh`
     - `tests/scripts/linux/test_run.sh`
   - **Test script**: `scripts/test-example.sh`.

### Phase 9 – Documentation and Cleanup
1. **P9T1 – README updates**
   - Document new architecture, development setup with Volta & Bun, testing strategy, and usage notes.
   - Test script: `scripts/test-readme.sh` runs `markdownlint` if available (ignore failures if tool missing).

## Additional Guidance
- Prefer functional, modular design so each piece can be tested independently.
- Keep data in memory; no server or filesystem writes beyond user-triggered downloads.
- When creating new folders, place an `index.ts` that re-exports module contents for simpler imports.
- If a task is blocked by missing context, add an `AICODE-ASK` comment and describe the question in the report.
- Ensure all timestamps use UTC (`date -u`).

