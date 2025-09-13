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
2. **P0T2 – Directory scaffolding**
   - Create the `src` subfolders listed above with placeholder index files.
   - Add `tests/setup.ts` if needed for Vitest.
   - Test script: reuse `scripts/test-toolchain.sh`.

### Phase 1 – RDF & CSV Utility Layer
1. **P1T1 – RDF helpers** (`src/lib/rdf.ts`)
   - Wrap an RDF library (e.g., `n3`) to create quads, serialize to Turtle and N‑Quads.
   - Expose `createDataset`, `serializeDataset`.
   - Unit tests: `tests/lib/rdf.test.ts`.
   - Test script: `scripts/test-lib-rdf.sh` → `bun test tests/lib/rdf.test.ts`.
2. **P1T2 – CSV helpers** (`src/lib/csv.ts`)
   - Utilities for loading CSV in browser and applying transformations.
   - Implement async `loadCsv` returning array of records.
   - Unit tests: `tests/lib/csv.test.ts`.
   - Test script: `scripts/test-lib-csv.sh`.

### Phase 2 – Draw.io Parsing
1. **P2T1 – XML to internal model** (`src/drawio/parser.ts`)
   - Parse Draw.io XML into `Individual` and `Arrow` objects modeled after Python `DrawIOXMLTree`.
   - Provide `parseDrawio(xml: string): MappingModel`.
   - Include necessary namespace handling and IRI creation.
   - Tests: `tests/drawio/parser.test.ts` with sample diagram fixtures.
   - Test script: `scripts/test-drawio-parser.sh`.
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
1. **P4T1 – Mapping model** (`src/mapping/model.ts`)
   - Define interfaces for mapping instructions derived from Draw.io (`TriplesMap`, `PredicateObjectMap`, etc.).
   - Include helper to build mapping model from Phase 2 output.
   - Tests: `tests/mapping/model.test.ts`.
   - Test script: `scripts/test-mapping-model.sh`.
2. **P4T2 – CSV → RDF triple conversion** (`src/mapping/mapper.ts`)
   - Implement function `mapCsvToRdf(mapping: MappingModel, csv: Record[]): Dataset`.
   - Integrate preprocessing from Phase 3.
   - First round post‑processing hooks (`applyInitialPostprocess(dataset)` in `src/postprocess/initial.ts`).
   - Tests: `tests/mapping/mapper.test.ts` with small fixture CSV and Draw.io pair.
   - Test script: `scripts/test-mapping-mapper.sh`.
   - **Relevant Files**:
     - **Python Scripts**: `map_rml.py`
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`, `tests/scripts/linux/test_rg_1-429.sh`, `tests/scripts/linux/test_run.sh`
     - **Referenced Files**:
       - RML files generated by `map_schema.py` (e.g., `gbad/schema/generic/generic/generic_schema.rml`)
       - `gbad/mapping/target/`

### Phase 5 – Postprocessing Modules
1. **P5T1 – Initial postprocessing** (`src/postprocess/initial.ts`)
   - Port existing postprocessing steps from `map_rml.py` (e.g., URI normalization).
   - Tests: `tests/postprocess/initial.test.ts`.
   - Test script: `scripts/test-postprocess-initial.sh`.
   - **Relevant Files**:
     - **Python Scripts**: `map_rml.py` (contains `postprocess` function)
     - **Integration Tests**: `tests/scripts/linux/test_add.sh`, `tests/scripts/linux/test_auth.sh`, `tests/scripts/linux/test_rg_1-429.sh`, `tests/scripts/linux/test_run.sh`
2. **P5T2 – Merge-stage postprocessing** (`src/postprocess/merge.ts`)
   - New second-round transformations to apply after graph merging.
   - Stub out with `AICODE-TODO` where rules are unspecified.
   - Tests: `tests/postprocess/merge.test.ts`.
   - Test script: `scripts/test-postprocess-merge.sh`.

### Phase 6 – Graph Merge & Named Graph Handling
1. **P6T1 – Merge utility** (`src/merge/merger.ts`)
   - Combine multiple datasets, generate named graph URIs (sha256 → ni → uuid as in `merge_graphs.py`).
   - Apply `applyMergePostprocess` from Phase 5.
   - Output N‑Quads string.
   - Tests: `tests/merge/merger.test.ts`.
   - Test script: `scripts/test-merge-merger.sh`.
   - **Relevant Files**:
     - **Python Scripts**: `merge_graphs.py`
     - **Integration Tests**: `tests/scripts/linux/test_run.sh`
     - **Referenced Files**:
       - Output of `map_rml.py` (e.g., `.ttl` or `.nt` files in `gbad/mapping/target/`)
       - `tests/data/store.nq`

### Phase 7 – React User Interface
1. **P7T1 – File upload & pipeline orchestration** (`src/ui/App.tsx` + components)
   - UI to upload Draw.io and CSV pairs, execute pipeline entirely in browser, show status and download merged graph.
   - Use React hooks, maintain pipeline state.
   - Tests: `tests/ui/app.test.tsx` with Vitest + React Testing Library.
   - Test script: `scripts/test-ui-app.sh`.

### Phase 8 – Integration & Example Workflow
1. **P8T1 – End-to-end example**
   - Add `examples/` with sample Draw.io & CSV, plus script `scripts/run-example.sh` demonstrating full pipeline.
   - Test script: `scripts/test-example.sh` to run `bun scripts/run-example.sh` and verify output file exists.

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

