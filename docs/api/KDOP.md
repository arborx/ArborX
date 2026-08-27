# KDOP

**Header:** `<ArborX_KDOP.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`KDOP<DIM, k, Coordinate>` is a *k*-Discrete Oriented Polytope (k-DOP) bounding volume. A k-DOP is a convex polytope defined by projecting the geometry onto a fixed set of *k/2* slab directions and recording the minimum and maximum extents along each direction. k-DOPs are tighter than axis-aligned bounding boxes and can lead to fewer false positives during tree traversal.

An empty k-DOP is one where `_min_values[i] > _max_values[i]` for all directions. The default constructor produces such an empty k-DOP.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <int DIM, int k, typename Coordinate = float>
struct KDOP : public Details::KDOP_Directions<DIM, k, Coordinate>
{
    static constexpr int n_directions = /* k/2 */;

    Kokkos::Array<Coordinate, n_directions> _min_values;
    Kokkos::Array<Coordinate, n_directions> _max_values;

    KOKKOS_FUNCTION KDOP();          // empty k-DOP
    KOKKOS_FUNCTION explicit operator Box<DIM, Coordinate>() const;
};

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Description |
|-----------|-------------|
| `DIM` | Spatial dimension (2 or 3). |
| `k` | Total number of bounding planes (must be even). Supported values depend on `DIM` (see table below). |
| `Coordinate` | Scalar coordinate type (default: `float`). |

---

## Supported (DIM, k) combinations

| DIM | k | n_directions | Description |
|-----|---|:---:|-------------|
| 2 | 4 | 2 | Axis-aligned bounding rectangle (equivalent to `Box<2>`) |
| 2 | 8 | 4 | 2-D 8-DOP (adds diagonal slabs) |
| 3 | 6 | 3 | Axis-aligned bounding box (equivalent to `Box<3>`) |
| 3 | 14 | 7 | 3-D 14-DOP (adds face diagonals) |
| 3 | 18 | 9 | 3-D 18-DOP (adds edge diagonals) |
| 3 | 26 | 13 | 3-D 26-DOP (adds face+edge diagonals) |

---

## Data members

| Member | Description |
|--------|-------------|
| `_min_values` | Array of length `n_directions` holding the minimum projection value along each direction. |
| `_max_values` | Array of length `n_directions` holding the maximum projection value along each direction. |

---

## Constructors

### Default constructor

```cpp
KOKKOS_FUNCTION KDOP();
```

Initialises all `_min_values[i]` to the largest representable `Coordinate` value and all `_max_values[i]` to the smallest representable value, producing an empty k-DOP.

---

## Conversion

```cpp
KOKKOS_FUNCTION explicit operator Box<DIM, Coordinate>() const;
```

Converts the k-DOP to an axis-aligned bounding box by extracting the axis-aligned component from the first `DIM` slab directions.

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<KDOP<DIM,k,C>>` | `DIM` |
| `tag_t<KDOP<DIM,k,C>>` | `KDOPTag` |
| `coordinate_type_t<KDOP<DIM,k,C>>` | `C` |

---

## Notes

- k-DOPs can be used as both stored primitives and bounding volumes in `BoundingVolumeHierarchy` by setting the `BoundingVolume` template parameter explicitly.
- Higher `k` values provide tighter fits but require more memory and more overlap test work per node.
- `expand` and `intersects` overloads are provided for all supported (DIM, k) combinations.

---

## Example

```cpp
#include <ArborX_KDOP.hpp>

// 3-D 26-DOP around a single point
ArborX::Experimental::KDOP<3, 26> kdop;
ArborX::Point<3> p{1.f, 2.f, 3.f};
ArborX::expand(kdop, p);

// Convert to AABB
ArborX::Box<3> aabb = static_cast<ArborX::Box<3>>(kdop);
```

---

## See also

- [`Box`](Box.md)
- [`GeometryTraits`](GeometryTraits.md)
- [Geometries requirements](GeometriesRequirements.md)
