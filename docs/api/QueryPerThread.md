# Query per thread (PerThread)

**Header:** `<ArborX_LinearBVH.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

By default, `BoundingVolumeHierarchy::query` launches a Kokkos parallel dispatch where **one thread handles all predicates**. The `PerThread` tag enables a different mode: a single predicate is evaluated from within an already-running device kernel, one call per thread.

This is useful when tree queries must be embedded inside larger parallel algorithms where the Kokkos execution space is already occupied—for example, inside a `Kokkos::parallel_for` body.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

struct PerThread {};

} // namespace ArborX::Experimental
```

### `BoundingVolumeHierarchy::query` (PerThread overload)

```cpp
template <typename Predicate, typename Callback>
KOKKOS_FUNCTION
void BoundingVolumeHierarchy::query(Experimental::PerThread,
                                    Predicate const& predicate,
                                    Callback  const& callback) const;
```

| Parameter | Description |
|-----------|-------------|
| *(tag)* | `Experimental::PerThread{}` — selects this overload. |
| `predicate` | A single predicate (e.g., `intersects(geometry)` or `nearest(geometry, k)`). |
| `callback` | Called for every match; must satisfy the **inline callback** requirements (see [`Callbacks`](Callbacks.md)). |

This function is marked `KOKKOS_FUNCTION` and is therefore callable from both host and device code.

---

## Notes

- Unlike the parallel-query overload, this function does **not** accept a `TraversalPolicy`; there is no predicate sorting when issuing queries one-by-one.
- The callback is invoked synchronously in the calling thread; no additional parallelism is introduced.
- The tree (`BoundingVolumeHierarchy`) must reside in a memory space accessible from the calling execution space.
- This overload is intended for use inside `KOKKOS_LAMBDA` or `KOKKOS_FUNCTION` bodies. It should **not** be called from the host when a parallel kernel is not already running, as the traversal will execute entirely on the host in that case.
- There is no output view mechanism for this overload; all results must be captured directly inside the callback.

---

## Example

```cpp
#include <ArborX_LinearBVH.hpp>

// Build the tree on the host/device
ArborX::BoundingVolumeHierarchy bvh(space, primitives);

// Use inside a parallel kernel — one query per thread
Kokkos::parallel_for(
    "my_kernel",
    Kokkos::RangePolicy(space, 0, num_queries),
    KOKKOS_LAMBDA(int i) {
        bvh.query(
            ArborX::Experimental::PerThread{},
            ArborX::intersects(query_boxes(i)),
            [&](auto const& /*predicate*/, auto const& value) {
                // process match for thread i
                results(i, ...) = value;
            });
    });
```

---

## See also

- [`BoundingVolumeHierarchy`](BVH.md)
- [`Callbacks`](Callbacks.md)
- [`TraversalPolicy`](TraversalPolicy.md)
