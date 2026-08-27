# Triangle

**Header:** `<ArborX_Triangle.hpp>`  
**Namespace:** `ArborX`

---

## Overview

`Triangle<DIM, Coordinate>` represents a triangle in *DIM*-dimensional space defined by three vertices `a`, `b`, `c`.

---

## Synopsis

```cpp
namespace ArborX {

template <int DIM, class Coordinate = float>
struct Triangle {
    Point<DIM, Coordinate> a;
    Point<DIM, Coordinate> b;
    Point<DIM, Coordinate> c;
};

// Deduction guides
template <int DIM, class Coordinate>
KOKKOS_DEDUCTION_GUIDE Triangle(Point<DIM,Coordinate>,
                                 Point<DIM,Coordinate>,
                                 Point<DIM,Coordinate>)
    -> Triangle<DIM, Coordinate>;

template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE Triangle(T const (&)[N], T const (&)[N], T const (&)[N])
    -> Triangle<N, T>;

} // namespace ArborX
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIM` | — | Spatial dimension. |
| `Coordinate` | `float` | Scalar coordinate type. |

---

## Data members

| Member | Description |
|--------|-------------|
| `a` | First vertex. |
| `b` | Second vertex. |
| `c` | Third vertex. |

All three vertices are public; `Triangle` is an aggregate.

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Triangle<DIM,C>>` | `DIM` |
| `tag_t<Triangle<DIM,C>>` | `TriangleTag` |
| `coordinate_type_t<Triangle<DIM,C>>` | `C` |

---

## Notes

- ArborX does not verify that the three points are non-collinear. Degenerate (zero-area) triangles may cause undefined behaviour in some algorithms.
- For 3-D problems, the most common usage is `Triangle<3, float>` or `Triangle<3, double>`.
- Triangle–ray intersection is supported via `ArborX::Experimental::Ray` (see [`Ray`](Ray.md)).

---

## Example

```cpp
ArborX::Triangle<3> tri{
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f}
};
```

---

## See also

- [`Point`](Point.md)
- [`Ray`](Ray.md)
- [`GeometryTraits`](GeometryTraits.md)
