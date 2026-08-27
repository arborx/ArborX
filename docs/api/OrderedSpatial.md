# Ordered Spatial Queries (`ordered_intersects`)

**Header:** `<detail/ArborX_Predicates.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

An **ordered spatial** predicate is a variant of a spatial intersection predicate that additionally **sorts results by ascending distance** from the query geometry to each matching primitive. All primitives that intersect the query geometry are returned, but they are delivered to the callback in nearest-first order.

This is useful when only the closest few results are needed, or when downstream processing benefits from an already-sorted result set, without having to use a `nearest` predicate with a fixed `k`.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <typename Geometry>
struct OrderedSpatial {
    using Tag = Details::OrderedSpatialPredicateTag;

    Geometry _geometry;

    template <class OtherGeometry>
    KOKKOS_FUNCTION auto distance(OtherGeometry const& other) const;
};

template <typename Geometry>
KOKKOS_INLINE_FUNCTION
OrderedSpatial<Geometry> ordered_intersects(Geometry const& geometry);

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Description |
|-----------|-------------|
| `Geometry` | Any geometry type for which `ArborX::distance` and `ArborX::intersects` are defined. |

### `ordered_intersects`

```cpp
template <typename Geometry>
KOKKOS_INLINE_FUNCTION
ArborX::Experimental::OrderedSpatial<Geometry>
ordered_intersects(Geometry const& geometry);
```

Creates an `OrderedSpatial` predicate wrapping `geometry`.

---

## Callback requirements

The callback for an ordered spatial predicate follows the same rules as for a regular spatial predicate:

```cpp
struct MyCallback {
    template <typename Predicate, typename Value>
    KOKKOS_FUNCTION void operator()(Predicate const& predicate,
                                    Value     const& value) const;
    // optionally return CallbackTreeTraversalControl
};
```

The callback may return `ArborX::CallbackTreeTraversalControl::early_exit` to stop traversal once a sufficient number of results has been collected (e.g., the first `k` neighbours). This makes ordered-spatial queries a flexible alternative to `nearest` queries when `k` is not known in advance.

---

## Comparison with other predicates

| Predicate | Results | Ordered? | Early exit? |
|-----------|---------|----------|-------------|
| `intersects(g)` | All intersecting primitives | No | Yes |
| `nearest(g, k)` | Exactly `k` nearest primitives | Yes (by construction) | No |
| `ordered_intersects(g)` | All intersecting primitives | Yes (ascending distance) | Yes |

---

## Notes

- `ordered_intersects` uses `ArborX::distance` to rank candidates; the same geometry support requirements as `nearest` apply.
- The tag `OrderedSpatialPredicateTag` distinguishes this predicate type from `SpatialPredicateTag` at compile time.
- Sorting is performed internally by the traversal engine; no additional post-processing by the user is required.
- Callback return type `void` or `CallbackTreeTraversalControl` are both valid.

---

## Example

```cpp
#include <detail/ArborX_Predicates.hpp>
#include <ArborX_LinearBVH.hpp>

ArborX::BoundingVolumeHierarchy bvh(space, primitives);

// Collect up to the first 5 primitives that intersect a box, sorted by distance
Kokkos::View<int*, MemorySpace> nearest5("nearest5", 5);
Kokkos::View<int , MemorySpace> count("count");

bvh.query(
    space,
    ArborX::Experimental::attach_indices(
        Kokkos::View<ArborX::Experimental::OrderedSpatial<ArborX::Box<3>>*,
                     MemorySpace>{...}),
    KOKKOS_LAMBDA(auto const& predicate, auto const& value)
        -> ArborX::CallbackTreeTraversalControl {
        int idx = Kokkos::atomic_fetch_inc(&count());
        if (idx < 5) {
            nearest5(idx) = value.index;
            return ArborX::CallbackTreeTraversalControl::normal_continuation;
        }
        return ArborX::CallbackTreeTraversalControl::early_exit;
    });
```

---

## See also

- [`Callbacks`](Callbacks.md)
- [`attach_indices`](attach_indices.md)
- [`BoundingVolumeHierarchy`](BVH.md)
- [Predicates overview](Predicates.md)
