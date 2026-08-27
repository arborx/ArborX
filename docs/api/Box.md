# Box

**Header:** `<ArborX_Box.hpp>`  
**Namespace:** `ArborX`

---

## Overview

`Box<DIM, Coordinate>` is an axis-aligned bounding box (AABB) in *DIM*-dimensional space, defined by a minimum corner and a maximum corner. It is the default bounding volume used by `BoundingVolumeHierarchy`.

An empty box is one where, for every dimension, `min_corner[d] > max_corner[d]`. The default constructor produces such an empty box.

---

## Synopsis

```cpp
namespace ArborX {

template <int DIM, class Coordinate = float>
struct Box {
    // Constructs an empty box
    KOKKOS_FUNCTION constexpr Box();

    // Construct from explicit corners
    KOKKOS_FUNCTION constexpr Box(Point<DIM, Coordinate> const& min_corner,
                                   Point<DIM, Coordinate> const& max_corner);

    KOKKOS_FUNCTION constexpr auto&       minCorner();
    KOKKOS_FUNCTION constexpr auto const& minCorner() const;
    KOKKOS_FUNCTION constexpr auto&       maxCorner();
    KOKKOS_FUNCTION constexpr auto const& maxCorner() const;

private:
    Point<DIM, Coordinate> _min_corner;
    Point<DIM, Coordinate> _max_corner;
};

} // namespace ArborX
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIM` | — | Spatial dimension. |
| `Coordinate` | `float` | Scalar coordinate type. |

---

## Constructors

### Default constructor

```cpp
KOKKOS_FUNCTION constexpr Box();
```

Creates an *empty* box. `_min_corner[d]` is set to the largest finite representable value of `Coordinate` and `_max_corner[d]` to the smallest finite value. This ensures that `expand(empty_box, point)` always yields a box containing only that point.

### Corner constructor

```cpp
KOKKOS_FUNCTION constexpr Box(Point<DIM, Coordinate> const& min_corner,
                               Point<DIM, Coordinate> const& max_corner);
```

Constructs a box with the given corners. No validation is performed (callers are responsible for ensuring `min <= max`).

---

## Member functions

### `minCorner` / `maxCorner`

```cpp
KOKKOS_FUNCTION constexpr Point<DIM, Coordinate>&       minCorner();
KOKKOS_FUNCTION constexpr Point<DIM, Coordinate> const& minCorner() const;
KOKKOS_FUNCTION constexpr Point<DIM, Coordinate>&       maxCorner();
KOKKOS_FUNCTION constexpr Point<DIM, Coordinate> const& maxCorner() const;
```

Access the minimum and maximum corners.

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Box<DIM,C>>` | `DIM` |
| `tag_t<Box<DIM,C>>` | `BoxTag` |
| `coordinate_type_t<Box<DIM,C>>` | `C` |

---

## Notes

- `Box` serves as the `Kokkos::reduction_identity` for expand-reduce operations; its identity value is the empty box.
- `Box` is a Kokkos reducer identity type, so it can be used as the accumulator in `Kokkos::parallel_reduce` with `expand` as the binary operator.

---

## Example

```cpp
ArborX::Box<3> b{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}};
auto& lo = b.minCorner();  // {0,0,0}
auto& hi = b.maxCorner();  // {1,1,1}

// Default (empty) box
ArborX::Box<3> empty;
```

---

## See also

- [`Point`](Point.md)
- [`Sphere`](Sphere.md)
- [`GeometryTraits`](GeometryTraits.md)
- [Geometries requirements](GeometriesRequirements.md)
