# Geometries requirements

**Headers:** Various (see individual geometry pages)  
**Namespace:** `ArborX`

---

## Overview

For a user-defined type to be accepted as a **geometry** by ArborX, three conditions must be satisfied simultaneously:

1. **`GeometryTraits` specialisations** — `dimension`, `tag`, and `coordinate_type` must be specialised.
2. **Algorithm overloads** — the required free functions must be provided for the combination of geometries involved.
3. **Device accessibility** — the type and its relevant operations must be usable from device code (`KOKKOS_FUNCTION`).

---

## 1. GeometryTraits requirements

Specialise the three trait templates in `ArborX::GeometryTraits`:

```cpp
// Required for any geometry
template <> struct ArborX::GeometryTraits::dimension<MyGeom> {
    static constexpr int value = /* positive integer */;
};
template <> struct ArborX::GeometryTraits::tag<MyGeom> {
    using type = /* one of the recognised tag types */;
};
template <> struct ArborX::GeometryTraits::coordinate_type<MyGeom> {
    using type = /* arithmetic type, e.g. float or double */;
};
```

See [`GeometryTraits`](GeometryTraits.md) for the full list of recognised tags.

---

## 2. Algorithm requirements

The algorithms that ArborX needs depend on how the geometry is used.

### As a **stored primitive** in the tree

The tree must be able to compute a bounding box for each stored primitive:

| Algorithm | Signature | Notes |
|-----------|-----------|-------|
| `expand` | `KOKKOS_FUNCTION void expand(Box& box, MyGeom const& g)` | Expand `box` to contain `g`. |
| `centroid` (optional) | `KOKKOS_FUNCTION auto returnCentroid(MyGeom const& g) -> Point<DIM>` | Used for space-filling-curve ordering. Default falls back to bounding-box centre. |

### As a **predicate geometry** in an `intersects` query

| Algorithm | Signature |
|-----------|-----------|
| `intersects` | `KOKKOS_FUNCTION bool intersects(MyGeom const& a, OtherGeom const& b)` |

### As a **predicate geometry** in a `nearest` or `ordered_intersects` query

| Algorithm | Signature |
|-----------|-----------|
| `distance` | `KOKKOS_FUNCTION auto distance(MyGeom const& a, OtherGeom const& b) -> Scalar` |

Overloads must be provided for all (predicate geometry, bounding-volume) pairs that can arise during traversal.

---

## 3. Device accessibility

All types and free functions used in device-side traversal must be annotated with `KOKKOS_FUNCTION` (or `KOKKOS_INLINE_FUNCTION`). Structs that contain only trivially copyable members are automatically device-compatible.

---

## Compile-time validation

Calling the following function will trigger `static_assert` errors for any missing trait:

```cpp
ArborX::GeometryTraits::check_valid_geometry_traits(my_geometry);
```

The tree also validates access traits and space-filling-curve compatibility during construction.

---

## Minimal example

```cpp
// 2D rectangle used as a primitive
struct Rect { float x0, x1, y0, y1; };

template <> struct ArborX::GeometryTraits::dimension<Rect>       { static constexpr int value = 2; };
template <> struct ArborX::GeometryTraits::tag<Rect>             { using type = ArborX::GeometryTraits::BoxTag; };
template <> struct ArborX::GeometryTraits::coordinate_type<Rect> { using type = float; };

// expand: required for tree construction
KOKKOS_FUNCTION void expand(ArborX::Box<2>& b, Rect const& r) {
    expand(b, ArborX::Point<2>{r.x0, r.y0});
    expand(b, ArborX::Point<2>{r.x1, r.y1});
}

// intersects: required for spatial queries
KOKKOS_FUNCTION bool intersects(Rect const& a, Rect const& b) {
    return a.x0 <= b.x1 && a.x1 >= b.x0 &&
           a.y0 <= b.y1 && a.y1 >= b.y0;
}
```

---

## Summary table

| Use case | Required traits | Required algorithms |
|----------|----------------|---------------------|
| Primitive stored in BVH | `dimension`, `tag`, `coordinate_type` | `expand` |
| `intersects` predicate | All three traits | `intersects(predGeom, primBV)` |
| `nearest` / `ordered_intersects` predicate | All three traits | `distance(predGeom, primBV)` |
| Space-filling-curve ordering | All three traits | `expand` (centroid optional) |

---

## See also

- [`GeometryTraits`](GeometryTraits.md)
- [Point](Point.md), [Box](Box.md), [Sphere](Sphere.md), [Triangle](Triangle.md), [Ray](Ray.md), [Segment](Segment.md), [Tetrahedron](Tetrahedron.md), [KDOP](KDOP.md)
