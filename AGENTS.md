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

### Phase 3 – CSV Preprocessing
1. **P3T1 – Custom preprocessing functions** (`src/csv/preprocess.ts`)
   - Port logic from `map_schema.py` (column splits, field logic).
   - Functions should accept raw CSV records and output transformed records.
   - Mark any domain‑specific behavior needing clarification with `AICODE-ASK`.
   - Tests: `tests/csv/preprocess.test.ts`.
   - Test script: `scripts/test-csv-preprocess.sh`.

### Phase 4 – Mapping Engine (RML removal)
1. **P4T1 – Mapping model** (`src/mapping/model.ts`)
   - Define interfaces for mapping instructions derived from Draw.io (`TriplesMap`, `PredicateObjectMap`, etc.).
   - Include helper to build mapping model from Phase 2 output.
   - Tests: `tests/mapping/model.test.ts`.
   - Test script: `scripts/test-mapping-model.sh`.
2. **P4T2 – CSV → RDF triple conversion** (`src/mapping/mapper.ts`) <!-- reviewed -->
   - **Goal:** Implement the core mapping engine that replaces the legacy `rmlmapper.jar`. This function will take a mapping model and preprocessed CSV data and generate an RDF dataset. This is not a direct port of Python code but a new implementation of a mapping engine.
   - **Function Signature:** `mapCsvToRdf(mapping: MappingModel, csvData: Record<string, any>[]): Dataset`
   - **Inputs:**
     - `mapping`: The `MappingModel` instance built in P4T1. This model contains a list of `TriplesMap` objects, each defining how to generate a subject and its properties.
     - `csvData`: An array of records from the preprocessed CSV file (output of P3T1). Each record is an object where keys are column headers.
   - **Core Implementation Steps:**
     1. Initialize an empty RDF/JS-compliant `Dataset` (e.g., from the `n3` library).
     2. Iterate over each `record` in the `csvData` array.
     3. For each `record`, iterate through each `triplesMap` in the `mapping` model.
     4. **Subject Generation:** Construct the subject IRI for the current `triplesMap` using its `subjectTemplate` (e.g., `KB/Agent/{HEADING}`) by replacing placeholders like `{HEADING}` with values from the current `record`.
     5. **Type Generation:** For the generated subject, add `rdf:type` triples for every class specified in the `triplesMap.rdfTypes` array.
     6. **Predicate-Object Generation:** Iterate through the `predicateObjectMaps` array of the current `triplesMap`. For each `pom`:
        - **Conditional Triple Creation:** The creation of this triple is conditional. Implement a check that mimics the legacy `fno_map_value_unless_isnull` logic: only proceed to generate the object if the source column(s) referenced in the `pom`'s object map are present and not empty in the current `record`.
        - **Object Generation:** Determine the object's value and type based on the `pom.objectMap`:
          - If it's a `reference`, get the value from `record[objectMap.value]`. This will be an `rdf:Literal`.
          - If it's a `template`, construct an IRI by replacing placeholders with values from the `record`. This will be an `rdf:NamedNode`.
          - If it's a `parentTriplesMap`, this indicates a join. The object is the subject of another `TriplesMap`. You will need to find the corresponding `TriplesMap` in the `mapping` model and generate its subject IRI using the *same* current `record`.
          - If it's a `constant`, use the provided value directly as a `NamedNode` or `Literal`.
        - Add the complete `(subject, predicate, object)` triple to the dataset.
   - **Post-processing Integration:** This task includes the *first round* of post-processing, which should be applied to the dataset *after* the main mapping loop is complete.
     - Create a function `applyInitialPostprocess(dataset: Dataset)` in `src/postprocess/initial.ts` and call it from `mapCsvToRdf`.
     - Implement the logic from the legacy `remove_shorter_duplicate_labels` function (`map_rml.py`):
       1. Find all subjects of type `rico:RecordSet`.
       2. For each, if it has exactly two `rdfs:label` properties, find its related `add:CurrentReferenceCode` identifier.
       3. Extract the reference code string from the identifier's label (e.g., get "C 119" from "C 119 (Current Reference Code)").
       4. If both `rdfs:label`s of the `RecordSet` start with this reference code, remove the triple containing the shorter label from the dataset.
     - `AICODE-NOTE`: Other post-processing logic from the legacy scripts, such as merging supplementary graphs, is out of scope for this task and will be handled in Phase 6.
   - **Testing:**
     - Unit tests in `tests/mapping/mapper.test.ts` should cover:
       - Basic triple generation from a simple mapping.
       - Correct IRI generation from templates.
       - Correct literal generation from references.
       - The "not null" conditional logic (a triple is not generated for an empty source column).
       - The `remove_shorter_duplicate_labels` post-processing logic.
   - **Test Script:** `scripts/test-mapping-mapper.sh`.

### Phase 5 – Postprocessing Modules
1. **P5T1 – Initial postprocessing** (`src/postprocess/initial.ts`)
   - Port existing postprocessing steps from `map_rml.py` (e.g., URI normalization).
   - Tests: `tests/postprocess/initial.test.ts`.
   - Test script: `scripts/test-postprocess-initial.sh`.
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

