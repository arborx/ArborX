# DefaultIndexableGetter

**Header:** `<detail/ArborX_IndexableGetter.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`DefaultIndexableGetter` is the default *indexable getter* used by `BoundingVolumeHierarchy`. An indexable getter is a callable that maps a stored **value** to a **geometry** that the BVH can index (i.e. compute a bounding volume for).

`DefaultIndexableGetter` handles two common cases transparently:

1. The value is already a geometry — it is returned as-is.
2. The value is a `PairValueIndex<Geometry, Index>` (as produced by `attach_indices`) — the `value` field (a geometry) is extracted and returned.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

struct DefaultIndexableGetter {
    DefaultIndexableGetter() = default;

    // Case 1: plain geometry (not a PairValueIndex)
    template <typename Geometry>
    KOKKOS_FUNCTION Geometry const& operator()(Geometry const& geometry) const;

    template <typename Geometry>
      requires(!Details::is_pair_value_index_v<std::decay_t<Geometry>>)
    KOKKOS_FUNCTION Geometry operator()(Geometry&& geometry) const;

    // Case 2: PairValueIndex — extract the geometry
    template <typename Geometry, typename Index>
    KOKKOS_FUNCTION Geometry const&
    operator()(PairValueIndex<Geometry, Index> const& pair) const;

    template <typename Geometry, typename Index>
    KOKKOS_FUNCTION Geometry
    operator()(PairValueIndex<Geometry, Index>&& pair) const;
};

} // namespace ArborX::Experimental
```

### Return value

| Input type | Return type |
|------------|-------------|
| `Geometry` (not `PairValueIndex`) | `Geometry const&` or `Geometry` (forwarded) |
| `PairValueIndex<Geometry, Index>` | `Geometry const&` or `Geometry` |

---

## Custom indexable getters

When the stored value type is not a geometry and not a `PairValueIndex`, a custom indexable getter must be provided. It must be a callable that:

- accepts a `const&` (or `&&`) reference to the value type, and
- returns a type for which `GeometryTraits` are defined.

```cpp
struct MyIndexableGetter {
    KOKKOS_FUNCTION
    auto const& operator()(MyRecord const& r) const {
        return r.bounding_box;  // extract the geometry
    }
};

ArborX::BoundingVolumeHierarchy<MemorySpace, MyRecord, MyIndexableGetter>
    bvh(space, values, MyIndexableGetter{});
```

---

## Notes

- `DefaultIndexableGetter` is used as the third template parameter of `BoundingVolumeHierarchy` when none is specified.
- The indexable getter is called once per stored value during tree **construction** and once per candidate value during **traversal** to retrieve the geometry for intersection or distance tests.
- The getter must be copy-constructible and callable from device code (`KOKKOS_FUNCTION`).

---

## `BoundingVolumeHierarchy` template signature (excerpt)

```cpp
template <
    typename MemorySpace,
    typename Value,
    typename IndexableGetter = Experimental::DefaultIndexableGetter,
    typename BoundingVolume  = /* derived from IndexableGetter and Value */>
class BoundingVolumeHierarchy { ... };
```

---

## See also

- [`attach_indices`](attach_indices.md)
- [`BoundingVolumeHierarchy`](BVH.md)
- [`GeometryTraits`](GeometryTraits.md)
