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
#ifndef ARBORX_DETAILS_GEOMETRY_EXPAND_HPP
#define ARBORX_DETAILS_GEOMETRY_EXPAND_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_MinMax.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct expand;
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

namespace Dispatch
{

using namespace GeometryTraits;

// expand a box to include a point
template <typename Box, typename Point>
struct expand<BoxTag, PointTag, Box, Point>
{
  KOKKOS_FUNCTION static void apply(Box &box, Point const &point)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int DIM = dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] = min(box.minCorner()[d], point[d]);
      box.maxCorner()[d] = max(box.maxCorner()[d], point[d]);
    }
  }
};

// expand a box to include a box
template <typename Box1, typename Box2>
struct expand<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static void apply(Box1 &box, Box2 const &other)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int DIM = dimension_v<Box1>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] = min(box.minCorner()[d], other.minCorner()[d]);
      box.maxCorner()[d] = max(box.maxCorner()[d], other.maxCorner()[d]);
    }
  }
};

// expand a box to include a sphere
template <typename Box, typename Sphere>
struct expand<BoxTag, SphereTag, Box, Sphere>
{
  KOKKOS_FUNCTION static void apply(Box &box, Sphere const &sphere)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int DIM = dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] =
          min(box.minCorner()[d], sphere.centroid()[d] - sphere.radius());
      box.maxCorner()[d] =
          max(box.maxCorner()[d], sphere.centroid()[d] + sphere.radius());
    }
  }
};

// expand a box to include a triangle
template <typename Box, typename Triangle>
struct expand<BoxTag, TriangleTag, Box, Triangle>
{
  KOKKOS_FUNCTION static void apply(Box &box, Triangle const &triangle)
  {
    Details::expand(box, triangle.a);
    Details::expand(box, triangle.b);
    Details::expand(box, triangle.c);
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
