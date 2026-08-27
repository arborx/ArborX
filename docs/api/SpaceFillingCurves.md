# Space-Filling Curves

**Header:** `<detail/ArborX_SpaceFillingCurves.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

Space-filling curves (SFCs) map multi-dimensional points to a one-dimensional linear order that preserves spatial locality. ArborX uses an SFC during **tree construction** to sort the input geometries into a coherent traversal order, which improves cache efficiency and BVH quality.

Two built-in SFC functors are provided:

| Type | Code width | Best for |
|------|-----------|---------|
| `Morton32` | 32-bit unsigned int | Most practical workloads |
| `Morton64` | 64-bit unsigned long long | Large datasets or high-precision coordinates |

Both are available in `ArborX::Experimental`.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

struct Morton32 {
    template <typename Box, typename Geometry>
    KOKKOS_FUNCTION auto operator()(Box const& scene_bounding_box,
                                    Geometry const& geometry) const
        -> unsigned int;
};

struct Morton64 {
    template <typename Box, typename Geometry>
    KOKKOS_FUNCTION auto operator()(Box const& scene_bounding_box,
                                    Geometry const& geometry) const
        -> unsigned long long;
};

} // namespace ArborX::Experimental
```

### Template parameters (operator())

| Parameter | Constraint | Description |
|-----------|-----------|-------------|
| `Box` | `GeometryTraits::is_box_v<Box>` | Bounding box of the entire scene. Used to normalise coordinates into `[0, 1]^DIM`. |
| `Geometry` | Any geometry with a centroid | The geometry whose Morton code is computed. The centroid is extracted via `Details::returnCentroid`. |

### Return value

An unsigned integer (`uint32_t` for `Morton32`, `uint64_t` for `Morton64`) representing the position of the geometry along the Morton space-filling curve.

---

## Custom space-filling curves

A user-defined SFC can be passed as the fourth template parameter (or constructor argument) to `BoundingVolumeHierarchy`. It must satisfy:

```
curve(scene_bounding_box, point) -> T
curve(scene_bounding_box, box)   -> T
```

where `T` is either `unsigned int` or `unsigned long long`, and both overloads must return the **same** type. The library performs a compile-time check via `check_valid_space_filling_curve<DIM>(curve)`.

```cpp
struct MySFC {
    KOKKOS_FUNCTION
    unsigned int operator()(ArborX::Box<3> const& bounds,
                             ArborX::Point<3> const& p) const { ... }
    KOKKOS_FUNCTION
    unsigned int operator()(ArborX::Box<3> const& bounds,
                             ArborX::Box<3>  const& b) const { ... }
};
```

---

## How it works

1. The scene bounding box is computed from all stored geometries.
2. Each geometry's centroid is projected into normalised `[0, 1]^DIM` coordinates relative to the scene bounding box.
3. The normalised point is interleaved bitwise (Morton encoding) to produce the linear order key.
4. The geometries are sorted by this key before building the hierarchy.

---

## Notes

- `Morton64` is the **default** SFC for `BoundingVolumeHierarchy`. It provides better ordering for large datasets or when coordinate precision is critical.
- Switching to `Morton32` reduces memory and may be marginally faster for small datasets.
- SFC ordering also governs predicate sorting during `query` (see [`TraversalPolicy`](TraversalPolicy.md)).
- The SFC is applied to centroids only; the actual geometry shape does not affect the ordering key.

---

## Example

```cpp
#include <detail/ArborX_SpaceFillingCurves.hpp>
#include <ArborX_LinearBVH.hpp>

// Build with default Morton64
ArborX::BoundingVolumeHierarchy bvh64(space, primitives);

// Build with Morton32 (slightly lower memory, same API)
ArborX::BoundingVolumeHierarchy bvh32(
    space, primitives,
    ArborX::Experimental::DefaultIndexableGetter{},
    ArborX::Experimental::Morton32{});
```

---

## See also

- [`TraversalPolicy`](TraversalPolicy.md)
- [`BoundingVolumeHierarchy`](BVH.md)
- [`GeometryTraits`](GeometryTraits.md)
