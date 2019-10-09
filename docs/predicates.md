# `ArborX::nearest`, `ArborX::intersects`
Defined in header [`<ArborX_Predicates.hpp>`](https://github.com/arborx/ArborX/blob/master/src/details/ArborX_Predicates.hpp)

## Spatial queries
[ArborX::intersects](#arborxintersects)
## Nearest neighbors queries
[ArborX::nearest](#arborxnearest)


## See also
[`ArborX::BVH<DeviceType>::query()`](https://github.com/arborx/ArborX/blob/docs/docs/bounding_volume_hierarchy.md#arborxbvhdevicetypequery)

# `ArborX::nearest`
```C++
template <typename Geometry>
KOKKOS_FUNCTION unspecified nearest(Geometry const& geometry, std::size_t k) noexcept;
```
Generate a nearest predicate to perform a k-nearest neighbors search with `ArborX::BVH<DeviceType>::query()`.
## Template parameter(s)
`Geometry`
: The geometry type, e.g. `ArborX::Point`, `ArborX::Box`, or `ArborX::Sphere`.
## Parameter(s)
`geometry`
: The geometry object from which distance is calculated.  
`k`
: The number of primitives to search for.
## Example
```C++
auto nearest_five_to_origin_pred = ArborX::nearest(ArborX::Point{0, 0, 0}, 5);
```

# `ArborX::intersects`
```C++
template <typename Geometry>
KOKKOS_FUNCTION unspecified intersects(Geometry const& geometry) noexcept;
```
Generate a spatial predicate to perform a search with `ArborX::BVH<DeviceType>::query()`.
## Template parameter(s)
`Geometry`
: The geometry type, e.g. `ArborX::Point`, `ArborX::Box`, or `ArborX::Sphere`.
## Parameter(s)
`geometry`
: The geometry object.
## Example
```C++
auto intersects_with_unit_sphere_pred = ArborX::intersects(ArborX::Sphere{{{0, 0, 0}}, 1});
```
