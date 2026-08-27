# Tetrahedron

**Header:** `<ArborX_Tetrahedron.hpp>`  
**Namespace:** `ArborX::ExperimentalHyperGeometry`

---

## Overview

`Tetrahedron<Coordinate>` represents a tetrahedron in 3-D space defined by four vertices `a`, `b`, `c`, `d`.

---

## Synopsis

```cpp
namespace ArborX::ExperimentalHyperGeometry {

template <class Coordinate = float>
struct Tetrahedron {
    Point<3, Coordinate> a;
    Point<3, Coordinate> b;
    Point<3, Coordinate> c;
    Point<3, Coordinate> d;
};

// Deduction guides
template <class Coordinate>
KOKKOS_DEDUCTION_GUIDE
Tetrahedron(Point<3,Coordinate>, Point<3,Coordinate>,
            Point<3,Coordinate>, Point<3,Coordinate>)
    -> Tetrahedron<Coordinate>;

template <typename T>
KOKKOS_DEDUCTION_GUIDE
Tetrahedron(T const (&)[3], T const (&)[3],
            T const (&)[3], T const (&)[3])
    -> Tetrahedron<T>;

} // namespace ArborX::ExperimentalHyperGeometry
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Coordinate` | `float` | Scalar coordinate type. |

`Tetrahedron` is always 3-D.

---

## Data members

| Member | Description |
|--------|-------------|
| `a` | First vertex. |
| `b` | Second vertex. |
| `c` | Third vertex. |
| `d` | Fourth vertex. |

All members are public; `Tetrahedron` is an aggregate.

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Tetrahedron<C>>` | 3 |
| `tag_t<Tetrahedron<C>>` | `TetrahedronTag` |
| `coordinate_type_t<Tetrahedron<C>>` | `C` |

---

## Notes

- `Tetrahedron` lives in `ArborX::ExperimentalHyperGeometry` because it is still under active development.
- ArborX does not verify that the four points are non-coplanar. Degenerate tetrahedra may cause undefined behaviour in some algorithms.
- `Tetrahedron` is always 3-D; there is no `DIM` template parameter.

---

## Example

```cpp
ArborX::ExperimentalHyperGeometry::Tetrahedron<float> tet{
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f},
    {0.f, 1.f, 0.f},
    {0.f, 0.f, 1.f}
};
```

---

## See also

- [`Point`](Point.md)
- [`Triangle`](Triangle.md)
- [`GeometryTraits`](GeometryTraits.md)
