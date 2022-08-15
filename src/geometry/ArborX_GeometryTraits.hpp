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

#include <Kokkos_Core.hpp>

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

template <typename Geometry>
struct dimension
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Geometry>
struct tag
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Geometry>
using DimensionNotSpecializedArchetypeAlias =
    typename dimension<Geometry>::not_specialized;

template <typename Geometry>
using TagNotSpecializedArchetypeAlias = typename tag<Geometry>::not_specialized;

template <typename Geometry>
using DimensionArchetypeAlias = decltype(dimension<Geometry>::value);

template <typename Geometry>
using TagArchetypeAlias = typename tag<Geometry>::type;

template <typename Geometry>
struct is_point
    : Kokkos::is_detected_exact<PointTag, TagArchetypeAlias, Geometry>
{};

template <typename Geometry>
struct is_box : Kokkos::is_detected_exact<BoxTag, TagArchetypeAlias, Geometry>
{};

template <typename Geometry>
struct is_sphere
    : Kokkos::is_detected_exact<SphereTag, TagArchetypeAlias, Geometry>
{};

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
      Kokkos::is_detected<DimensionArchetypeAlias, Geometry>{},
      "GeometryTraits::dimension<Geometry> must define 'value' member type");
  static_assert(
      std::is_integral<
          Kokkos::detected_t<DimensionArchetypeAlias, Geometry>>{} &&
          GeometryTraits::dimension<Geometry>::value > 0,
      "GeometryTraits::dimension<Geometry>::value must be a positive integral");

  static_assert(Kokkos::is_detected<TagArchetypeAlias, Geometry>{},
                "GeometryTraits::tag<Geometry> must define 'type' member type");
  using Tag = Kokkos::detected_t<TagArchetypeAlias, Geometry>;
  static_assert(std::is_same<Tag, PointTag>{} || std::is_same<Tag, BoxTag>{} ||
                    std::is_same<Tag, SphereTag>{},
                "GeometryTraits::tag<Geometry>::type must be PointTag, BoxTag "
                "or SphereTag");
}

} // namespace GeometryTraits

} // namespace ArborX

#endif
