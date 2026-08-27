# MinimumSpanningTree

**Header:** `<ArborX_MinimumSpanningTree.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`MinimumSpanningTree` computes the **Euclidean Minimum Spanning Tree (EMST)** of a point cloud, or optionally the MST of the **mutual-reachability graph** used by HDBSCAN. The algorithm is based on Borůvka's algorithm accelerated with a BVH.

Given `n` input points, the result is `n-1` weighted edges that connect all points with minimum total weight (Euclidean distance by default).

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <class MemorySpace,
          Details::BoruvkaMode Mode = Details::BoruvkaMode::MST>
struct MinimumSpanningTree
{
    using memory_space = MemorySpace;

    Kokkos::View<WeightedEdge*, MemorySpace> edges;

    // Internal (HDBSCAN) views — not used for plain MST queries
    Kokkos::View<int*,   MemorySpace> dendrogram_parents;
    Kokkos::View<float*, MemorySpace> dendrogram_parent_heights;

    // Constructor
    template <class ExecutionSpace, class Primitives>
    MinimumSpanningTree(ExecutionSpace const& space,
                        Primitives     const& primitives,
                        int k = 1);
};

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MemorySpace` | — | Kokkos memory space where the output edges are stored. |
| `Mode` | `BoruvkaMode::MST` | Algorithm variant. Use `BoruvkaMode::MST` for the standard EMST. `BoruvkaMode::HDBSCAN` is used internally by the HDBSCAN clustering algorithm. |

---

## Constructor

```cpp
template <class ExecutionSpace, class Primitives>
MinimumSpanningTree(ExecutionSpace const& space,
                    Primitives     const& primitives,
                    int k = 1);
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `space` | Kokkos execution space for all parallel operations. |
| `primitives` | Point cloud satisfying `AccessTraits`. Elements must satisfy `GeometryTraits::is_point_v`. Must contain at least 1 point. |
| `k` | Core-distance parameter for mutual reachability distance. `k = 1` (default) gives the standard Euclidean MST. `k > 1` switches to mutual-reachability distance, making the MST sensitive to local density (used for HDBSCAN). |

### Postcondition

After construction, `edges` contains exactly `primitives.size() - 1` edges.

---

## `WeightedEdge`

```cpp
namespace ArborX::Experimental {

struct WeightedEdge {
    int   source;
    int   target;
    float weight;
};

} // namespace ArborX::Experimental
```

Each edge records the indices of two connected points and the distance (weight) between them.

---

## Algorithm

`MinimumSpanningTree` uses **Borůvka's parallel algorithm**:

1. A BVH is constructed over the input points (using `attach_indices`).
2. In each iteration, the shortest outgoing edge for each connected component is found in parallel using BVH traversal.
3. Components are merged along found edges.
4. Steps 2–3 repeat until a single spanning tree remains (`O(log n)` iterations).

When `k > 1`, core distances are precomputed via `k`-nearest-neighbour queries and used to define mutual-reachability weights.

---

## Notes

- The output `edges` are **not** sorted by weight; sort them if needed for downstream algorithms.
- `MinimumSpanningTree` with `Mode = BoruvkaMode::HDBSCAN` additionally populates `dendrogram_parents` and `dendrogram_parent_heights`; use [`Dendrogram`](Dendrogram.md) for a cleaner interface.
- Points must be accessible from `ExecutionSpace`.
- The weight type is always `float`. Use `double` coordinates only when higher positional accuracy is needed.

---

## Example

```cpp
#include <ArborX_MinimumSpanningTree.hpp>

// points: Kokkos::View<ArborX::Point<3>*, MemorySpace>
ArborX::Experimental::MinimumSpanningTree<MemorySpace> mst(
    exec_space, points);

// Access the MST edges
auto& edges = mst.edges;  // size: n - 1
Kokkos::parallel_for(edges.size(), KOKKOS_LAMBDA(int i) {
    auto e = edges(i);
    // e.source, e.target, e.weight
});
```

---

## See also

- [`Dendrogram`](Dendrogram.md)
- [`attach_indices`](attach_indices.md)
- [`BoundingVolumeHierarchy`](BVH.md)
