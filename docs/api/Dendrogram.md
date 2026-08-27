# Dendrogram

**Header:** `<ArborX_Dendrogram.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`Dendrogram` represents a hierarchical clustering tree (dendrogram) built from a set of weighted edges — typically the output of `MinimumSpanningTree`. Each internal node of the dendrogram corresponds to a merge event at a specific distance level.

The dendrogram is stored as two parallel arrays:
- `_parents` — parent index for each leaf and internal node.
- `_parent_heights` — merge height (distance) for each internal node.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

enum class DendrogramImplementation {
    BORUVKA,
    UNION_FIND
};

template <typename MemorySpace>
struct Dendrogram
{
    using memory_space = MemorySpace;

    Kokkos::View<int*,   MemorySpace> _parents;
    Kokkos::View<float*, MemorySpace> _parent_heights;

    // Construct from pre-computed parents / heights
    Dendrogram(Kokkos::View<int*,   MemorySpace> parents,
               Kokkos::View<float*, MemorySpace> parent_heights);

    // Construct from MST edges (union-find algorithm)
    template <typename ExecutionSpace>
    Dendrogram(ExecutionSpace const& exec_space,
               Kokkos::View<Experimental::WeightedEdge*, MemorySpace> edges);
};

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Description |
|-----------|-------------|
| `MemorySpace` | Kokkos memory space where the dendrogram arrays are stored. |

---

## Constructors

### From raw arrays

```cpp
Dendrogram(Kokkos::View<int*,   MemorySpace> parents,
           Kokkos::View<float*, MemorySpace> parent_heights);
```

Constructs a dendrogram from precomputed parent and height arrays. Both views are stored by reference (no copy).

### From MST edges (union-find)

```cpp
template <typename ExecutionSpace>
Dendrogram(ExecutionSpace const& exec_space,
           Kokkos::View<Experimental::WeightedEdge*, MemorySpace> edges);
```

Builds a dendrogram from a set of `n-1` weighted MST edges (as produced by `MinimumSpanningTree`). The algorithm:

1. Splits edges into weights and unweighted connectivity.
2. Sorts edges by ascending weight.
3. Applies a union-find algorithm to assign parent indices to all `2n-1` nodes (n leaves + n-1 internal nodes).

| Parameter | Description |
|-----------|-------------|
| `exec_space` | Kokkos execution space. |
| `edges` | View of `n-1` `WeightedEdge` objects. The view is consumed (sorted in-place). |

---

## Data members

| Member | Size | Description |
|--------|------|-------------|
| `_parents` | `2n - 1` | `_parents[i]` is the index of the parent of node `i`. The root node is its own parent (or a sentinel value). |
| `_parent_heights` | `n - 1` | `_parent_heights[i]` is the merge height (distance) at internal node `i`. |

Node indexing convention:
- Nodes `0 .. n-1` are leaves (correspond to input points).
- Nodes `n .. 2n-2` are internal nodes (merge events).

---

## Notes

- `Dendrogram` built from `MinimumSpanningTree` with `Mode = BoruvkaMode::HDBSCAN` can be obtained directly from `mst.dendrogram_parents` and `mst.dendrogram_parent_heights` without constructing a second `Dendrogram` object.
- The union-find constructor sorts `edges` in-place; do not rely on the original order after construction.
- `_parent_heights` has size `n-1`, not `2n-1`. Only internal nodes have a defined height.

---

## Example

```cpp
#include <ArborX_MinimumSpanningTree.hpp>
#include <ArborX_Dendrogram.hpp>

// Compute MST
ArborX::Experimental::MinimumSpanningTree<MemorySpace> mst(
    exec_space, points);

// Build dendrogram from MST edges
ArborX::Experimental::Dendrogram<MemorySpace> dendrogram(
    exec_space, mst.edges);

// Access dendrogram structure
auto& parents        = dendrogram._parents;         // size: 2n-1
auto& parent_heights = dendrogram._parent_heights;  // size: n-1
```

---

## See also

- [`MinimumSpanningTree`](MinimumSpanningTree.md)
- [`BoundingVolumeHierarchy`](BVH.md)
