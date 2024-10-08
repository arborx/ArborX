/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_GEOMETRY_VALID_HPP
#define ARBORX_DETAILS_GEOMETRY_VALID_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag, typename Geometry>
struct isValid;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool isValid(Geometry const &geometry)
{
  return Dispatch::isValid<typename GeometryTraits::tag_t<Geometry>,
                           Geometry>::apply(geometry);
}

namespace Dispatch
{

using namespace GeometryTraits;

// isValid point
template <typename Point>
struct isValid<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &p)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (!Kokkos::isfinite(p[d]))
        return false;
    return true;
  }
};

// isValid box
template <typename Box>
struct isValid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
      if (r_d <= 0 || !Kokkos::isfinite(r_d))
        return false;
    }
    return true;
  }
};

// isValid sphere
template <typename Sphere>
struct isValid<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &s)
  {
    return Details::isValid(s.centroid()) && Kokkos::isfinite(s.radius()) &&
           (s.radius() >= 0.);
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
