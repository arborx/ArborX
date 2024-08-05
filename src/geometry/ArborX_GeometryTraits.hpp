/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_GEOMETRY_TRAITS_HPP
#define ARBORX_GEOMETRY_TRAITS_HPP

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX
{

namespace GeometryTraits
{

template <typename Geometry>
struct dimension
{
  using not_specialized = void; // tag to detect existence of a specialization
};
template <typename Geometry>
inline constexpr int dimension_v = dimension<std::remove_cv_t<Geometry>>::value;

struct not_specialized
{};

template <typename Geometry>
struct tag
{
  using type = not_specialized;
};
template <typename Geometry>
using tag_t = typename tag<std::remove_cv_t<Geometry>>::type;

template <typename Geometry>
struct coordinate_type
{
  using type = not_specialized;
};
template <typename Geometry>
using coordinate_type_t =
    typename coordinate_type<std::remove_cv_t<Geometry>>::type;

// clang-format off
#define DEFINE_GEOMETRY(name, name_tag)                                        \
  struct name_tag{};                                                           \
  template <typename Geometry>                                                 \
  struct is_##name : std::is_same<tag_t<Geometry>, name_tag>{};              \
  template <typename Geometry>                                                 \
  inline constexpr bool is_##name##_v = is_##name<Geometry>::value
// clang-format on

DEFINE_GEOMETRY(point, PointTag);
DEFINE_GEOMETRY(box, BoxTag);
DEFINE_GEOMETRY(sphere, SphereTag);
DEFINE_GEOMETRY(triangle, TriangleTag);
DEFINE_GEOMETRY(kdop, KDOPTag);
DEFINE_GEOMETRY(obb, OBBTag);
DEFINE_GEOMETRY(tetrahedron, TetrahedronTag);
DEFINE_GEOMETRY(ray, RayTag);
#undef DEFINE_GEOMETRY

template <typename Geometry>
inline constexpr bool
    is_valid_geometry = (is_point_v<Geometry> || is_box_v<Geometry> ||
                         is_sphere_v<Geometry> || is_kdop_v<Geometry> ||
                         is_obb_v<Geometry> || is_triangle_v<Geometry> ||
                         is_tetrahedron_v<Geometry> || is_ray_v<Geometry>);

template <typename Geometry>
using DimensionNotSpecializedArchetypeAlias =
    typename dimension<Geometry>::not_specialized;

template <typename Geometry>
using TagNotSpecializedArchetypeAlias = typename tag<Geometry>::not_specialized;

template <typename Geometry>
using CoordinateNotSpecializedArchetypeAlias =
    typename coordinate_type<Geometry>::not_specialized;

template <typename Geometry>
using DimensionArchetypeAlias = decltype(dimension_v<Geometry>);

template <typename Geometry>
void check_valid_geometry_traits(Geometry const &)
{
  static_assert(
      !Kokkos::is_detected<DimensionNotSpecializedArchetypeAlias, Geometry>{},
      "Must specialize GeometryTraits::dimension<Geometry>");
  static_assert(
      !Kokkos::is_detected<TagNotSpecializedArchetypeAlias, Geometry>{},
      "Must specialize GeometryTraits::tag<Geometry>");
  static_assert(
      !Kokkos::is_detected<CoordinateNotSpecializedArchetypeAlias, Geometry>{},
      "Must specialize GeometryTraits::coordinate_type<Geometry>");

  static_assert(
      Kokkos::is_detected<DimensionArchetypeAlias, Geometry>{},
      "GeometryTraits::dimension<Geometry> must define 'value' member type");
  static_assert(
      std::is_integral<
          Kokkos::detected_t<DimensionArchetypeAlias, Geometry>>{} &&
          GeometryTraits::dimension_v<Geometry> > 0,
      "GeometryTraits::dimension<Geometry>::value must be a positive integral");

  static_assert(!std::is_same_v<tag_t<Geometry>, not_specialized>,
                "GeometryTraits::tag<Geometry> must define 'type' member type");
  static_assert(is_valid_geometry<Geometry>,
                "GeometryTraits::tag<Geometry>::type must be PointTag, BoxTag, "
                "SphereTag, TriangleTag, KDOPTag, or RayTag");

  static_assert(!std::is_same_v<coordinate_type_t<Geometry>, not_specialized>,
                "GeometryTraits::coordinate_type<Geometry> must define 'type' "
                "member type");
  using Coordinate = coordinate_type_t<Geometry>;
  static_assert(
      std::is_arithmetic_v<Coordinate>,
      "GeometryTraits::coordinate_type<Geometry> must be an arithmetic type");
}

} // namespace GeometryTraits

} // namespace ArborX

#endif
