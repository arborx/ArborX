# Ray

**Header:** `<ArborX_Ray.hpp>`  
**Namespace:** `ArborX::Experimental`

---

## Overview

`Ray<Coordinate>` represents a 3-D ray defined by an **origin** and a **direction**. The direction vector is automatically normalised to unit length upon construction.

`Ray` is used as a query geometry for ray-tracing-style spatial queries against stored primitives (boxes, triangles, spheres).

---

## Synopsis

```cpp
namespace ArborX::Experimental {

template <typename Coordinate = float>
struct Ray {
    using Point  = ArborX::Point<3, Coordinate>;
    using Vector = ArborX::Details::Vector<3, Coordinate>;

    Point  _origin    = {};
    Vector _direction = {};

    KOKKOS_DEFAULTED_FUNCTION constexpr Ray() = default;

    KOKKOS_FUNCTION
    Ray(Point const& origin, Vector const& direction);
    // direction is normalised automatically

    KOKKOS_FUNCTION constexpr Point&        origin();
    KOKKOS_FUNCTION constexpr Point  const& origin()    const;
    KOKKOS_FUNCTION constexpr Vector const& direction() const;
};

} // namespace ArborX::Experimental
```

### Template parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Coordinate` | `float` | Scalar type for coordinates and direction components. |

Note: `Ray` is always 3-D.

---

## Constructors

### Default constructor

Produces a ray at the origin with a zero (invalid) direction. Intended for default-initialisation only.

### Value constructor

```cpp
KOKKOS_FUNCTION Ray(Point const& origin, Vector const& direction);
```

Creates a ray. The `direction` vector is normalised to unit length internally using higher precision (`double`) to reduce floating-point error in intersection tests.

---

## Member functions

| Function | Description |
|----------|-------------|
| `origin()` | Returns a reference to the origin point. |
| `direction()` | Returns a `const` reference to the unit-length direction vector. |

---

## GeometryTraits

| Trait | Value |
|-------|-------|
| `dimension_v<Ray<C>>` | 3 |
| `tag_t<Ray<C>>` | `RayTag` |
| `coordinate_type_t<Ray<C>>` | `C` |

---

## Notes

- `Ray` is always 3-D. There is no `DIM` template parameter.
- The direction normalisation uses `double` arithmetic even when `Coordinate = float` to avoid excessive rounding error.
- `Ray` supports intersection tests against `Box`, `Sphere`, and `Triangle` via `ArborX::intersects`.

---

## Example

```cpp
ArborX::Experimental::Ray<float> r{
    {0.f, 0.f, 0.f},   // origin
    {1.f, 0.f, 0.f}    // direction (will be normalised)
};

auto& o = r.origin();       // {0,0,0}
auto& d = r.direction();    // {1,0,0}
```

---

## See also

- [`Point`](Point.md)
- [`Triangle`](Triangle.md)
- [`GeometryTraits`](GeometryTraits.md)
