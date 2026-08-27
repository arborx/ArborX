# GeometryTraits

**Header:** `<ArborX_GeometryTraits.hpp>`  
**Namespace:** `ArborX::GeometryTraits`

---

## Overview

`GeometryTraits` is the customisation-point mechanism that ArborX uses to learn about geometry types. By specialising three trait templates for a type `Geometry`, the type becomes usable anywhere ArborX expects a geometry (tree construction, queries, algorithms).

---

## Trait templates

### `dimension<Geometry>`

```cpp
template <typename Geometry>
struct dimension {
    static constexpr int value; // must be a positive integer
};

template <typename Geometry>
inline constexpr int dimension_v = dimension<std::remove_cv_t<Geometry>>::value;
```

Specifies the spatial dimension of the geometry. Must be a positive integer.

---

### `tag<Geometry>`

```cpp
template <typename Geometry>
struct tag {
    using type = /* one of the tag types below */;
};

template <typename Geometry>
using tag_t = typename tag<std::remove_cv_t<Geometry>>::type;
```

Identifies the geometric shape. The `type` member must be one of:

| Tag | Shape |
|-----|-------|
| `GeometryTraits::PointTag` | Point |
| `GeometryTraits::BoxTag` | Axis-aligned bounding box |
| `GeometryTraits::SphereTag` | Sphere |
| `GeometryTraits::TriangleTag` | Triangle |
| `GeometryTraits::KDOPTag` | Discrete oriented polytope |
| `GeometryTraits::TetrahedronTag` | Tetrahedron |
| `GeometryTraits::RayTag` | Ray |
| `GeometryTraits::SegmentTag` | Line segment |
| `GeometryTraits::EllipsoidTag` | Ellipsoid |

---

### `coordinate_type<Geometry>`

```cpp
template <typename Geometry>
struct coordinate_type {
    using type = /* arithmetic scalar type */;
};

template <typename Geometry>
using coordinate_type_t =
    typename coordinate_type<std::remove_cv_t<Geometry>>::type;
```

Specifies the scalar arithmetic type used for coordinates (e.g., `float`, `double`).

---

## Helper traits

The following `is_*` traits and variables are automatically generated for each recognised tag:

```cpp
// Example for points:
template <typename Geometry>
struct is_point : std::is_same<tag_t<Geometry>, PointTag> {};

template <typename Geometry>
inline constexpr bool is_point_v = is_point<Geometry>::value;
```

Available: `is_point`, `is_box`, `is_sphere`, `is_triangle`, `is_kdop`, `is_tetrahedron`, `is_ray`, `is_segment`, `is_ellipsoid`, and their `_v` inline variable counterparts.

---

## Validation

```cpp
namespace ArborX::GeometryTraits {

template <typename Geometry>
void check_valid_geometry_traits(Geometry const&);

} // namespace ArborX::GeometryTraits
```

Calling `check_valid_geometry_traits(g)` triggers a series of `static_assert` checks that all three traits are properly specialised for `Geometry`. Useful as a compile-time diagnostic in user code.

---

## Example: registering a custom geometry

```cpp
// User-defined 2D bounding rectangle
struct MyRect {
    float x_min, x_max, y_min, y_max;
};

template <>
struct ArborX::GeometryTraits::dimension<MyRect> {
    static constexpr int value = 2;
};
template <>
struct ArborX::GeometryTraits::tag<MyRect> {
    using type = ArborX::GeometryTraits::BoxTag;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<MyRect> {
    using type = float;
};
```

After these specialisations, ArborX algorithms that operate on boxes will work with `MyRect` (provided an appropriate `expand` / `intersects` overload is also provided).

---

## Built-in geometries

ArborX ships the following types with pre-registered traits:

| Type | DIM | Tag | Coordinate |
|------|-----|-----|-----------|
| `ArborX::Point<DIM, C>` | `DIM` | `PointTag` | `C` |
| `ArborX::Box<DIM, C>` | `DIM` | `BoxTag` | `C` |
| `ArborX::Sphere<DIM, C>` | `DIM` | `SphereTag` | `C` |
| `ArborX::Triangle<DIM, C>` | `DIM` | `TriangleTag` | `C` |
| `ArborX::Experimental::KDOP<DIM, k, C>` | `DIM` | `KDOPTag` | `C` |
| `ArborX::ExperimentalHyperGeometry::Tetrahedron<C>` | 3 | `TetrahedronTag` | `C` |
| `ArborX::Experimental::Ray<C>` | 3 | `RayTag` | `C` |
| `ArborX::Experimental::Segment<DIM, C>` | `DIM` | `SegmentTag` | `C` |

---

## See also

- [Geometries requirements](GeometriesRequirements.md)
- [Point](Point.md), [Box](Box.md), [Sphere](Sphere.md), [Triangle](Triangle.md), [Ray](Ray.md), [Segment](Segment.md), [Tetrahedron](Tetrahedron.md), [KDOP](KDOP.md)
