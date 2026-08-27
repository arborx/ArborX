# TraversalPolicy

**Header:** `<detail/ArborX_TraversalPolicy.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`TraversalPolicy` is a configuration object passed to `BoundingVolumeHierarchy::query` that lets the user tune two aspects of the parallel tree traversal:

1. **Buffer size** — an optional upper bound on the number of results per query. When set correctly it allows the traversal to be performed in a single pass instead of two.
2. **Predicate sorting** — controls whether predicates are reordered along a space-filling curve before being dispatched. Sorting typically improves cache efficiency on GPU but adds overhead; it can be disabled for workloads where predicates are already well-ordered.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

struct TraversalPolicy {
    int  _buffer_size     = 0;
    bool _sort_predicates = true;

    TraversalPolicy& setBufferSize(int buffer_size);
    TraversalPolicy& setPredicateSorting(bool sort_predicates);
};

} // namespace ArborX::Experimental
```

---

## Member functions

### `setBufferSize`

```cpp
TraversalPolicy& setBufferSize(int buffer_size);
```

Sets the per-query result buffer size and returns `*this` (builder-style).

| Value | Behaviour |
|-------|-----------|
| `0` (default) | Buffer optimisation disabled. The traversal always performs two passes: the first counts results, the second writes them. |
| `> 0` (soft limit) | Preallocates `buffer_size` result slots per query. If the actual result count exceeds this for any query, the library falls back to a second pass automatically. |
| `< 0` (hard limit) | Preallocates `|buffer_size|` result slots per query. If the actual result count exceeds this for any query, an **exception is thrown** instead of falling back. |

A good buffer size estimate can eliminate the counting pass and roughly halve the traversal time. Use profiling to determine a reasonable upper bound.

### `setPredicateSorting`

```cpp
TraversalPolicy& setPredicateSorting(bool sort_predicates);
```

Enables or disables predicate reordering along a Morton space-filling curve before dispatching. Returns `*this`.

| Value | Behaviour |
|-------|-----------|
| `true` (default) | Predicates are sorted. Improves data locality on GPU; adds a sorting step. |
| `false` | Predicates are processed in the order they are supplied. Useful when they are already sorted, or when query count is very small. |

---

## Notes

- `TraversalPolicy` is default-constructible. The default configuration (buffer size 0, sorting enabled) is correct for all inputs and is a safe starting point.
- Both setters return `*this`, so they can be chained:
  ```cpp
  auto policy = ArborX::Experimental::TraversalPolicy{}
                    .setBufferSize(32)
                    .setPredicateSorting(false);
  ```
- The sign convention for `buffer_size` encodes the **fallback policy**: positive values are *soft limits* (safe), negative values are *hard limits* (throw on overflow).
- If the hard-limit is exceeded, the thrown exception carries a diagnostic message indicating which query overflowed.

---

## Example

```cpp
#include <detail/ArborX_TraversalPolicy.hpp>
#include <ArborX_LinearBVH.hpp>

ArborX::BoundingVolumeHierarchy bvh(space, primitives);

// Use a soft limit of 16 results per query; keep predicate sorting on
auto policy = ArborX::Experimental::TraversalPolicy{}.setBufferSize(16);

bvh.query(space, predicates, callback, policy);

// Hard limit: throw if any query produces more than 64 results
auto strict_policy = ArborX::Experimental::TraversalPolicy{}.setBufferSize(-64);

bvh.query(space, predicates, callback, strict_policy);
```

---

## See also

- [`BoundingVolumeHierarchy`](BVH.md)
- [`SpaceFillingCurves`](SpaceFillingCurves.md)
- [`Callbacks`](Callbacks.md)
