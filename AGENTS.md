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
2. **AICODE-TODO: P4T2.1 - Implement Basic Mapping Loop and Subject Generation** (`src/mapping/mapper.ts`)
   - **Goal**: Create the `mapCsvToRdf` function skeleton and implement the core iteration logic. This first step will focus on generating the primary resources (subjects) and their `rdf:type` declarations.
   - **Implementation**:
     - Create the `mapCsvToRdf` function.
     - Implement the main loops for iterating over `csvData` records and `mapping.TriplesMap` definitions.
     - For each record and triples map, generate the subject `NamedNode` by substituting placeholders in the `subjectTemplate` with values from the record.
     - Add the `rdf:type` triples for the subject using the `triplesMap.rdfTypes` array.
   - **Tests**: `tests/mapping/mapper.test.ts` should verify that subjects and their types are created correctly for a basic mapping.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

3. **AICODE-TODO: P4T2.2 - Add Object Generation for Literals and Constants** (`src/mapping/mapper.ts`)
   - **Goal**: Extend the mapper to handle the simplest object types: literals from CSV columns and constant values.
   - **Implementation**: In the `predicateObjectMaps` loop, add logic to handle object maps of type `reference` and `constant`.
     - For `reference`, create an `rdf:Literal` using the value from the specified CSV column in the current record.
     - For `constant`, create an `rdf:Literal` or `rdf:NamedNode` from the hardcoded value.
   - **Tests**: Extend `tests/mapping/mapper.test.ts` to test that properties with literal and constant objects are correctly generated.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

4. **AICODE-TODO: P4T2.3 - Implement Conditional Triple Generation** (`src/mapping/mapper.ts`)
   - **Goal**: Implement the "not null" condition to prevent creating triples when source data is missing.
   - **Implementation**: Wrap the predicate-object generation logic in a conditional check. If any source column for an object is empty, skip creating that triple.
   - **Tests**: `tests/mapping/mapper.test.ts` must include cases where source data is missing and verify that triples are *not* generated.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

5. **AICODE-TODO: P4T2.4 - Add Object Generation for Joins** (`src/mapping/mapper.ts`)
   - **Goal**: Implement support for linking between resources using `parentTriplesMap`.
   - **Implementation**: Add logic to handle `objectMap` of type `parentTriplesMap`, which requires looking up the target `TriplesMap` and generating its subject IRI using the same current CSV record.
   - **Tests**: `tests/mapping/mapper.test.ts` needs a test case with a mapping that uses `parentTriplesMap`.
   - **Test Script**: `scripts/test-mapping-mapper.sh`

6. **AICODE-TODO: P4T2.5 - Implement Initial Post-processing** (`src/postprocess/initial.ts`)
   - **Goal**: Implement the `remove_shorter_duplicate_labels` algorithm and integrate it into the pipeline.
   - **Implementation**: Create `applyInitialPostprocess` in `src/postprocess/initial.ts` with the required logic. Call it from `mapCsvToRdf` after the main mapping loop.
   - **Tests**: `tests/postprocess/initial.test.ts` should be created to specifically test the algorithm.
   - **Test Script**: `scripts/test-postprocess-initial.sh`

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

