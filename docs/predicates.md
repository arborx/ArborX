# `ArborX::nearest`, `ArborX::intersects`
Defined in header [`<ArborX_Predicates.hpp>`](https://github.com/arborx/ArborX/blob/master/src/details/ArborX_Predicates.hpp)

```C++
template <typename Geometry>
KOKKOS_FUNCTION unspecified nearest(Geometry const& geometry, std::size_t k) noexcept; // (1)

template <typename Geometry>
KOKKOS_FUNCTION unspecified intersects(Geometry const& geometry) noexcept; // (2)
```

## See also
[`ArborX::BVH<DeviceType>::query()`](https://github.com/dalg24/ArborX/blob/docs/docs/bounding_volume_hierarchy.md#arborxbvhdevicetypequery)

# `ArborX::nearest`

# `ArborX::intersects`