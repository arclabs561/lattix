---
status: accepted
date: 2026-07-09
scope: graph algorithms over duplicate triples
---

# Parallel Edge Semantics

Duplicate triples are stored. Default graph algorithms treat them as one
topological neighbor.

## Decision

Centrality, PageRank, personalized PageRank, HITS, Katz, eigenvector centrality,
closeness, betweenness, and shortest-path traversal use unique neighbor nodes
when traversing topology. Duplicate triples do not increase degree, path count,
rank mass, hub score, or authority score.

## Rationale

- A duplicated fact should not make an entity more important by accident.
- The default algorithms answer topology questions, not fact-frequency
  questions.
- Weighted behavior needs per-algorithm semantics and separate tests.

## Future Weighted Mode

Add weighted variants only when a caller needs multiplicity to affect scores.
That work should include golden tests on small multigraphs where the weighted
and topology-only answers intentionally differ.
