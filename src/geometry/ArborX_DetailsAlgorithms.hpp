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
#ifndef ARBORX_DETAILS_ALGORITHMS_HPP
#define ARBORX_DETAILS_ALGORITHMS_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Assert.hpp> // KOKKOS_ASSERT
#include <Kokkos_Macros.hpp>

namespace ArborX::Details
{

namespace Dispatch
{
template <typename Tag, typename Geometry>
struct equals;

template <typename Tag, typename Geometry>
struct isValid;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct distance;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct expand;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct intersects;

template <typename Tag, typename Geometry>
struct centroid;

} // namespace Dispatch

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Geometry const &l,
                                             Geometry const &r)
{
  return Dispatch::equals<typename GeometryTraits::tag_t<Geometry>,
                          Geometry>::apply(l, r);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool isValid(Geometry const &geometry)
{
  return Dispatch::isValid<typename GeometryTraits::tag_t<Geometry>,
                           Geometry>::apply(geometry);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto distance(Geometry1 const &geometry1,
                                     Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::distance<typename GeometryTraits::tag_t<Geometry1>,
                            typename GeometryTraits::tag_t<Geometry2>,
                            Geometry1, Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION void expand(Geometry1 &geometry1,
                                   Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  Dispatch::expand<typename GeometryTraits::tag_t<Geometry1>,
                   typename GeometryTraits::tag_t<Geometry2>, Geometry1,
                   Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Geometry1 const &geometry1,
                                                 Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::intersects<typename GeometryTraits::tag_t<Geometry1>,
                              typename GeometryTraits::tag_t<Geometry2>,
                              Geometry1, Geometry2>::apply(geometry1,
                                                           geometry2);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION decltype(auto) returnCentroid(Geometry const &geometry)
{
  return Dispatch::centroid<typename GeometryTraits::tag_t<Geometry>,
                            Geometry>::apply(geometry);
}

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point_v<Point> &&
                           GeometryTraits::is_box_v<Box>> * = nullptr>
KOKKOS_FUNCTION void translateAndScale(Point const &in, Point &out,
                                       Box const &ref)
{
  static_assert(GeometryTraits::dimension_v<Point> ==
                GeometryTraits::dimension_v<Box>);
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  for (int d = 0; d < DIM; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace ArborX::Details

#endif
