# Architect Report: P5T1 Task Enrichment
**Date:** 2025-09-13
**Agent:** Jules

## 1. Objective
The goal of this session was to act as a software architect and enrich one of the unreviewed tasks in the `AGENTS.md` implementation plan. The chosen task was required to be one of the more complex ones available.

## 2. Task Selection
Initially, I selected task `P2T1 – XML to internal model`. However, after receiving user feedback that `P2T1` and `P6T1` were already taken, I pivoted to a new task.

The final selected task was **`P5T1 – Initial postprocessing`**. This task was chosen for its complexity, which involves porting a non-trivial RDF post-processing pipeline from a Python script (`map_rml.py`) to a new TypeScript module (`src/postprocess/initial.ts`).

## 3. Analysis of `map_rml.py`
I conducted a thorough analysis of the `postprocess` function and its helper methods within `map_rml.py`. Key findings include:

*   **`remove_shorter_duplicate_labels`**: This is the main, active post-processing algorithm. It's a stateful graph transformation that removes `rdfs:label` triples based on a set of conditions related to `rico:RecordSet` resources and their identifiers. The logic is intricate and requires careful implementation to match the original behavior.
*   **`add_suppl_triples`**: This function merges additional RDF data from `.ttl` files located in the `gbad/schema/` directory. This will require a file-fetching mechanism in the new client-side architecture.
*   **Inactive Functions**: Two other functions, `remove_false_agentcontrolrelation` and `remove_false_authtp`, were found to be commented out in the Python script. Their original purpose was documented for potential future implementation.

## 4. Enrichment of `AGENTS.md`
Based on the analysis, I significantly enriched the description of task `P5T1` in `AGENTS.md`. The key enrichments are:

*   **Detailed Goal**: A clear, high-level goal for the task was established.
*   **Actionable Sub-tasks**: The task was broken down into three concrete `AICODE-TODO` items:
    1.  Implement the `remove_shorter_duplicate_labels` algorithm.
    2.  Implement the supplemental triple loading.
    3.  Stub out the inactive post-processing functions.
*   **Clear Contracts**: A specific TypeScript function signature for `applyInitialPostprocess` was provided to guide the implementing agent.
*   **Comprehensive Testing Strategy**: Detailed instructions were provided for unit testing each piece of functionality, including specific edge cases to cover.

## 5. Conclusion
Task `P5T1` in `AGENTS.md` is now fully specified and ready for an implementation agent to pick up. The detailed breakdown should make the execution straightforward and reduce ambiguity. The task has been marked as `<!-- reviewed -->`.
