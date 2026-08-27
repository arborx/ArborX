# Sphere

**Header:** `<ArborX_Sphere.hpp>`  
**Namespace:** `ArborX`

---

## Overview

`Sphere<DIM, Coordinate>` represents a *DIM*-dimensional ball defined by a centroid and a non-negative radius.

---

## Synopsis

```cpp
namespace ArborX {

template <int DIM, class Coordinate = float>
struct Sphere {
    KOKKOS_DEFAULTED_FUNCTION Sphere() = default;

    KOKKOS_FUNCTION constexpr Sphere(Point<DIM, Coordinate> const& centroid,
                                      Coordinate radius);

    KOKKOS_FUNCTION constexpr auto&       centroid();
    KOKKOS_FUNCTION constexpr auto const& centroid() const;
    KOKKOS_FUNCTION constexpr auto        radius() const;

private:
    Point<DIM, Coordinate> _centroid = {};
    Coordinate             _radius   = 0;
};

// Deduction guide
template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE Sphere(T const (&)[N], T) -> Sphere<N, T>;

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

Produces a zero-radius sphere at the origin.

### Value constructor

```cpp
KOKKOS_FUNCTION constexpr Sphere(Point<DIM, Coordinate> const& centroid,
                                   Coordinate radius);
```

Creates a sphere with the given centroid and radius.

---

## Member functions

| Function | Description |
|----------|-------------|
| `centroid()` | Returns a reference to the centroid `Point`. |
| `radius()` | Returns the radius (by value). |

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Sphere<DIM,C>>` | `DIM` |
| `tag_t<Sphere<DIM,C>>` | `SphereTag` |
| `coordinate_type_t<Sphere<DIM,C>>` | `C` |

---

## Example

```cpp
ArborX::Point<3> centre{1.f, 2.f, 3.f};
ArborX::Sphere<3> s{centre, 0.5f};

float r = s.radius();  // 0.5f
```

---

## See also

- [`Point`](Point.md)
- [`Box`](Box.md)
- [`GeometryTraits`](GeometryTraits.md)
