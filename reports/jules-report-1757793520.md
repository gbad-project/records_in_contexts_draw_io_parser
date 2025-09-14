# Architect's Report for Task P1T1: RDF Helpers

**Report ID:** `jules-report-1757793520`
**Date:** 2025-09-13
**Author:** Jules, AI Architect

## 1. Overview

This report provides the detailed technical specification for task **P1T1 – RDF helpers**. The goal of this task is to create a foundational module at `src/lib/rdf.ts` that provides a consistent, easy-to-use API for creating and managing RDF data. This module will wrap the `n3` library, abstracting away some of its complexities and ensuring that the rest of the application has a single, reliable source for RDF-related functionality.

## 2. Library Choice: `n3`

The `n3` library has been selected as the core dependency for this task.

*   **Rationale**: `n3` is a robust, performant, and feature-complete library for working with RDF in JavaScript. It includes tools for parsing, serializing, and querying RDF data, and its `DataFactory` and `Store` components provide a solid foundation for the helpers required by this project.
*   **Action for Developer**: The implementing agent must add `n3` and its corresponding type definitions to the project's dependencies by running:
    ```bash
    bun add n3 @types/n3
    ```

## 3. Module API: `src/lib/rdf.ts`

The `rdf.ts` module shall export the following functions and types.

### 3.1. Re-exported Types

For convenience and to establish `rdf.ts` as the canonical source for RDF types, the following types from the `n3` library should be imported and re-exported:

```typescript
import type { NamedNode, Literal, BlankNode, Quad, Store, DataFactory } from 'n3';
export type { NamedNode, Literal, BlankNode, Quad, Store, DataFactory };
```

### 3.2. Term Factory Functions

These functions are thin wrappers around `n3.DataFactory` to ensure a consistent method of term creation across the application.

```typescript
import { DataFactory } from 'n3';

// Creates a NamedNode.
export const namedNode = DataFactory.namedNode;

// Creates a Literal, with optional data type.
export const literal = DataFactory.literal;

// Creates a BlankNode.
export const blankNode = DataFactory.blankNode;

// Creates a Quad.
export const quad = DataFactory.quad;
```

### 3.3. Store Creation Function

A helper function to instantiate a new `n3.Store`.

```typescript
import { Store, Quad } from 'n3';

/**
 * Creates a new RDF store.
 * @param quads An optional array of quads to initialize the store with.
 * @returns A new n3.Store instance.
 */
export function createStore(quads?: Quad[]): Store {
  return new Store(quads);
}
```

### 3.4. Serialization Function

An asynchronous function to serialize an RDF store to a string.

```typescript
import { Store } from 'n3';
import { Writer } from 'n3';

/**
 * Serializes an RDF store to a string in the specified format.
 * @param store The n3.Store to serialize.
 * @param format The desired serialization format.
 * @returns A Promise that resolves with the serialized string.
 */
export async function serialize(store: Store, format: 'Turtle' | 'N-Quads' | 'N-Triples'): Promise<string> {
  return new Promise((resolve, reject) => {
    const writer = new Writer({ format });
    writer.addQuads(store.getQuads(null, null, null, null));
    writer.end((error, result) => {
      if (error) {
        return reject(error);
      }
      resolve(result);
    });
  });
}
```

## 4. Testing Strategy: `tests/lib/rdf.test.ts`

The following unit tests are required to ensure the module's correctness and reliability.

*   **Term Creation**: Verify that `namedNode`, `literal`, and `blankNode` return objects with the correct `termType` and `value`.
*   **Store Creation**:
    *   Test that `createStore()` with no arguments produces a store with `size` 0.
    *   Test that `createStore([...])` with an array of quads produces a store with the correct `size`.
*   **Serialization**:
    *   Create a sample store and serialize it to `'Turtle'`, `'N-Quads'`, and `'N-Triples'`. Verify that the output strings are syntactically correct and contain the expected data.
    *   Test that serializing an empty store produces a valid (and likely empty) string.
*   **Error Handling**:
    *   Verify that calling `serialize` with an invalid format (e.g., `'JSON-LD'`) throws an error.

This specification provides a complete guide for the implementation of task `P1T1`.
