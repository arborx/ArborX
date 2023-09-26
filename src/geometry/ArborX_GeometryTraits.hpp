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

struct PointTag
{};

struct BoxTag
{};

struct SphereTag
{};

struct TriangleTag
{};

struct KDOPTag
{};

template <typename Geometry>
struct dimension
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Geometry>
inline constexpr int dimension_v = dimension<Geometry>::value;

struct not_specialized
{};

template <typename Geometry>
struct tag
{
  using type = not_specialized;
};

template <typename Geometry>
struct coordinate_type
{
  using type = not_specialized;
};

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
struct is_point : std::is_same<typename tag<Geometry>::type, PointTag>
{};

template <typename Geometry>
struct is_box : std::is_same<typename tag<Geometry>::type, BoxTag>
{};

template <typename Geometry>
struct is_sphere : std::is_same<typename tag<Geometry>::type, SphereTag>
{};

template <typename Geometry>
struct is_triangle : std::is_same<typename tag<Geometry>::type, TriangleTag>
{};

template <typename Geometry>
KOKKOS_FUNCTION constexpr void check_valid_geometry_traits(Geometry const &)
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

  static_assert(
      !std::is_same<typename tag<Geometry>::type, not_specialized>::value,
      "GeometryTraits::tag<Geometry> must define 'type' member type");
  using Tag = typename tag<Geometry>::type;
  static_assert(std::is_same<Tag, PointTag>{} || std::is_same<Tag, BoxTag>{} ||
                    std::is_same<Tag, SphereTag>{} ||
                    std::is_same<Tag, TriangleTag>{} ||
                    std::is_same<Tag, KDOPTag>{},
                "GeometryTraits::tag<Geometry>::type must be PointTag, BoxTag, "
                "SphereTag, TriangleTag or KDOPTag");

  static_assert(!std::is_same<typename coordinate_type<Geometry>::type,
                              not_specialized>::value,
                "GeometryTraits::coordinate_type<Geometry> must define 'type' "
                "member type");
  using Coordinate = typename coordinate_type<Geometry>::type;
  static_assert(
      std::is_arithmetic_v<Coordinate>,
      "GeometryTraits::coordinate_type<Geometry> must be an arithmetic type");
}

} // namespace GeometryTraits

} // namespace ArborX

#endif
