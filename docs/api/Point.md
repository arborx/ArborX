# Point

**Header:** `<ArborX_Point.hpp>`  
**Namespace:** `ArborX`

---

## Overview

`Point<DIM, Coordinate>` is a simple fixed-size coordinate tuple representing a point in *DIM*-dimensional Euclidean space.

---

## Synopsis

```cpp
namespace ArborX {

template <int DIM, class Coordinate = float>
struct Point {
    static_assert(DIM > 0);

    KOKKOS_FUNCTION constexpr auto&       operator[](unsigned int i);
    KOKKOS_FUNCTION constexpr auto const& operator[](unsigned int i) const;

    Coordinate _coords[DIM] = {};
};

// Deduction guides
template <typename T, typename... Ts>
KOKKOS_DEDUCTION_GUIDE Point(T, Ts...) -> Point<sizeof...(Ts)+1, T>;

template <typename T, std::size_t N>
KOKKOS_DEDUCTION_GUIDE Point(T const (&)[N]) -> Point<N, T>;

} // namespace ArborX
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIM` | — | Spatial dimension. Must be `> 0`. |
| `Coordinate` | `float` | Scalar coordinate type (e.g., `float`, `double`). |

---

## Member functions

### `operator[]`

```cpp
KOKKOS_FUNCTION constexpr Coordinate&       operator[](unsigned int i);
KOKKOS_FUNCTION constexpr Coordinate const& operator[](unsigned int i) const;
```

Returns the `i`-th coordinate (zero-based). No bounds checking is performed.

---

## Data members

| Member | Description |
|--------|-------------|
| `_coords[DIM]` | Array of `DIM` coordinates, zero-initialised by default. |

---

## GeometryTraits

`Point` is pre-registered with `GeometryTraits`:

| Trait | Value |
|-------|-------|
| `dimension_v<Point<DIM,C>>` | `DIM` |
| `tag_t<Point<DIM,C>>` | `PointTag` |
| `coordinate_type_t<Point<DIM,C>>` | `C` |

---

## Notes

- `_coords` is zero-initialised by default, which enables use in `constexpr` contexts.
- The deduction guides allow natural construction: `ArborX::Point{1.0f, 2.0f, 3.0f}` deduces `Point<3, float>`.
- `Point` is a plain aggregate; it is trivially copyable and safe to use in device code.

---

## Example

```cpp
ArborX::Point<3> p{1.0f, 2.0f, 3.0f};
float x = p[0];  // 1.0f

ArborX::Point<2, double> q{0.5, 1.5};
```

---

## See also

- [`Box`](Box.md)
- [`Sphere`](Sphere.md)
- [`GeometryTraits`](GeometryTraits.md)
