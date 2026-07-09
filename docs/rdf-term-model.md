---
status: proposed
date: 2026-07-09
scope: RDF term representation
---

# RDF Term Model

`Triple` remains a string triple:

```rust
pub struct Triple {
    subject: EntityId,
    predicate: RelationType,
    object: EntityId,
}
```

The RDF readers and writers map RDF syntax onto that model:

- IRIs are stored as their IRI string.
- Blank nodes are stored as `_:label`.
- Literals are stored in N-Triples lexical form, including optional language or
  datatype suffix.
- Local non-IRI IDs are written as `urn:lattix:local:<percent-encoded-id>` and
  decoded on read.

This is an I/O compatibility layer. It is not a complete RDF term model because
the public graph API cannot distinguish a local string that looks like RDF
syntax from a caller intentionally supplying RDF syntax.

## Decision

Keep `Triple` as the primary graph model. If RDF callers need stronger typing,
add a parallel typed API such as `RdfTriple` or `Term` instead of changing
`Triple` internals first.

## Rationale

- Most graph algorithms only need node identity and edge labels.
- Replacing `Triple` internals would force RDF semantics onto non-RDF graph
  users.
- A parallel typed API can make RDF ambiguity explicit without breaking the core
  graph model immediately.

## Non-Goals

- No SPARQL engine.
- No RDF store semantics.
- No full RDF 1.2 triple-term support until typed terms are designed.

## Implementation Guardrail

Before adding typed public RDF terms, add tests covering:

- literals whose lexical values begin with quote-like or angle-bracket syntax,
- language tag and datatype normalization behavior,
- blank node identity across parse/write cycles,
- RDF 1.2 triple terms if they become supported.
