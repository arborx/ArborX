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
#ifndef ARBORX_DETAILS_GEOMETRY_EQUALS_HPP
#define ARBORX_DETAILS_GEOMETRY_EQUALS_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag, typename Geometry>
struct equals;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Geometry const &l,
                                             Geometry const &r)
{
  return Dispatch::equals<typename GeometryTraits::tag_t<Geometry>,
                          Geometry>::apply(l, r);
}

namespace Dispatch
{

using namespace GeometryTraits;

// equals point-point
template <typename Point>
struct equals<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &l, Point const &r)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (l[d] != r[d])
        return false;
    return true;
  }
};

// equals box-box
template <typename Box>
struct equals<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &l, Box const &r)
  {
    return Details::equals(l.minCorner(), r.minCorner()) &&
           Details::equals(l.maxCorner(), r.maxCorner());
  }
};

// equals sphere-sphere
template <typename Sphere>
struct equals<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &l, Sphere const &r)
  {
    return Details::equals(l.centroid(), r.centroid()) &&
           l.radius() == r.radius();
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
