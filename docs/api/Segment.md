# Segment

**Header:** `<ArborX_Segment.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`Segment<DIM, Coordinate>` represents a closed line segment in *DIM*-dimensional space defined by two endpoints `a` and `b`.

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <int DIM, typename Coordinate = float>
struct Segment {
    ArborX::Point<DIM, Coordinate> a;
    ArborX::Point<DIM, Coordinate> b;
};

// Deduction guides
template <int DIM, typename Coordinate>
KOKKOS_DEDUCTION_GUIDE Segment(Point<DIM,Coordinate>, Point<DIM,Coordinate>)
    -> Segment<DIM, Coordinate>;

template <int N, typename T>
KOKKOS_DEDUCTION_GUIDE Segment(T const (&)[N], T const (&)[N])
    -> Segment<N, T>;

} // namespace ArborX::Experimental
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
| `a` | First endpoint. |
| `b` | Second endpoint. |

Both members are public; `Segment` is an aggregate.

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Segment<DIM,C>>` | `DIM` |
| `tag_t<Segment<DIM,C>>` | `SegmentTag` |
| `coordinate_type_t<Segment<DIM,C>>` | `C` |

---

## Notes

- `Segment` lives in `ArborX::Experimental` because some algorithms involving segments are still under active development.
- A zero-length segment (`a == b`) is a valid degenerate case treated as a point.
- Distance from a segment to another geometry is computed via `ArborX::distance`.

---

## Example

```cpp
ArborX::Experimental::Segment<3> seg{
    {0.f, 0.f, 0.f},
    {1.f, 0.f, 0.f}
};
```

---

## See also

- [`Point`](Point.md)
- [`Triangle`](Triangle.md)
- [`GeometryTraits`](GeometryTraits.md)
