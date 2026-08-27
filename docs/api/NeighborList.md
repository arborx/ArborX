# Neighbor Lists

**Header:** `<detail/ArborX_NeighborList.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

ArborX provides two functions for computing **all pairs of points** within a given radius `r`:

| Function | Description |
|----------|-------------|
| `findHalfNeighborList` | For each pair `(i, j)` with `i < j`, only `i` is stored in `j`'s neighbour list (the pair is recorded once, under the **larger** index). |
| `findFullNeighborList` | Each pair is stored **twice**: `j` appears in `i`'s list and `i` appears in `j`'s list. |

Both functions output a **CSR (Compressed Sparse Row)** data structure consisting of an offset array and an index array.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <class ExecutionSpace, class Primitives,
          typename Coordinate, class Offsets, class Indices>
void findHalfNeighborList(ExecutionSpace const& space,
                          Primitives const& primitives,
                          Coordinate radius,
                          Offsets& offsets,
                          Indices& indices);

template <class ExecutionSpace, class Primitives,
          typename Coordinate, class Offsets, class Indices>
void findFullNeighborList(ExecutionSpace const& space,
                          Primitives const& primitives,
                          Coordinate radius,
                          Offsets& offsets,
                          Indices& indices);

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| `ExecutionSpace` | Kokkos execution space | Where the computation runs. |
| `Primitives` | Satisfies `AccessTraits`; elements are points | The set of points. |
| `Coordinate` | Arithmetic scalar | Radius type; must match the coordinate type of the points. |
| `Offsets` | `Kokkos::View<int*>` | Output: CSR offset array of length `n + 1`. |
| `Indices` | `Kokkos::View<int*>` | Output: CSR index array. |

### Parameters

| Parameter | Description |
|-----------|-------------|
| `space` | Kokkos execution space instance. |
| `primitives` | Point cloud satisfying `AccessTraits`. Elements must satisfy `GeometryTraits::is_point_v`. |
| `radius` | Search radius. Two points are neighbours if their Euclidean distance is `<= radius`. |
| `offsets` | Output offset array (resized by the function). |
| `indices` | Output index array (resized by the function). |

---

## Output format (CSR)

After a successful call:
- `offsets` has size `n + 1`, where `n = primitives.size()`.
- The neighbours of point `i` are `indices(offsets(i)), ..., indices(offsets(i+1)-1)`.

### Half list

For each pair `(i, j)` with `i < j`, only `i` is stored in `j`'s neighbour list. When iterating over the neighbours of point `i`, you will find only indices `j < i` (the smaller-index partner of each pair). This is the efficient representation for symmetric algorithms such as force calculations in molecular dynamics.

### Full list

Point `j` appears in the list of `i` and point `i` appears in the list of `j` (both triangles are stored). Useful when each point needs direct access to all its neighbours.

---

## Notes

- Both functions internally build a `BoundingVolumeHierarchy` over the input points (using `attach_indices`).
- The half-list traversal uses `Details::HalfTraversal`, which visits each pair exactly once without self-pairs.
- Self-pairs (`i == i`) are **never** stored.
- The `Offsets` and `Indices` views are **reallocated** by the function; pass uninitialised views.
- The coordinate type of the point cloud and the type of `radius` must be identical.
- Kokkos Profiling regions are emitted under `ArborX::Experimental::HalfNeighborList` and `ArborX::Experimental::FullNeighborList`.

---

## Example

```cpp
#include <detail/ArborX_NeighborList.hpp>

Kokkos::View<ArborX::Point<3>*, MemorySpace> points("points", n);
// ... fill points ...

Kokkos::View<int*, MemorySpace> offsets("offsets", 0);
Kokkos::View<int*, MemorySpace> indices("indices", 0);

ArborX::Experimental::findHalfNeighborList(
    exec_space, points, /*radius=*/ 0.1f, offsets, indices);

// Iterate over neighbours of point i (half list)
Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
    for (int k = offsets(i); k < offsets(i + 1); ++k) {
        int j = indices(k);  // j < i always
        // process pair (i, j)
    }
});
```

---

## See also

- [`attach_indices`](attach_indices.md)
- [`BoundingVolumeHierarchy`](BVH.md)
- [`Callbacks`](Callbacks.md)
