/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_GEOMETRY_CONVERT_HPP
#define ARBORX_DETAILS_GEOMETRY_CONVERT_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename TagFrom, typename TagTo, typename GeometryFrom,
          typename GeometryTo>
struct convert;
}

template <typename GeometryTo, typename GeometryFrom>
KOKKOS_INLINE_FUNCTION GeometryTo convert(GeometryFrom const &geometry)
{
  static_assert(GeometryTraits::dimension_v<GeometryFrom> ==
                GeometryTraits::dimension_v<GeometryTo>);
  return Dispatch::convert<typename GeometryTraits::tag_t<GeometryFrom>,
                           typename GeometryTraits::tag_t<GeometryTo>,
                           GeometryFrom, GeometryTo>::apply(geometry);
}

namespace Dispatch
{

using namespace GeometryTraits;

template <typename PointFrom, typename PointTo>
struct convert<PointTag, PointTag, PointFrom, PointTo>
{
  KOKKOS_FUNCTION static constexpr auto apply(PointFrom const &point)
  {
    PointTo converted;
    constexpr int DIM = GeometryTraits::dimension_v<PointFrom>;
    for (int d = 0; d < DIM; ++d)
      converted[d] = point[d];
    return converted;
  }
};

template <int DIM, typename Coordinate>
struct convert<PointTag, PointTag, Point<DIM, Coordinate>,
               Point<DIM, Coordinate>>
{
  KOKKOS_FUNCTION static constexpr auto
  apply(Point<DIM, Coordinate> const &point)
  {
    return point;
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
