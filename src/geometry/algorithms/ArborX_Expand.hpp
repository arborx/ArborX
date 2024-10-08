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

template <typename KDOP1, typename KDOP2>
struct expand<KDOPTag, KDOPTag, KDOP1, KDOP2>
{
  KOKKOS_FUNCTION static void apply(KDOP1 &that, KDOP2 const &other)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int n_directions = KDOP1::n_directions;
    static_assert(KDOP2::n_directions == n_directions);
    for (int i = 0; i < n_directions; ++i)
    {
      that._min_values[i] = min(that._min_values[i], other._min_values[i]);
      that._max_values[i] = max(that._max_values[i], other._max_values[i]);
    }
  }
};

template <typename KDOP, typename Point>
struct expand<KDOPTag, PointTag, KDOP, Point>
{
  KOKKOS_FUNCTION static void apply(KDOP &kdop, Point const &point)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    constexpr int n_directions = KDOP::n_directions;
    for (int i = 0; i < n_directions; ++i)
    {
      auto const &dir = kdop.directions()[i];
      auto proj_i = point[0] * dir[0];
      for (int d = 1; d < DIM; ++d)
        proj_i += point[d] * dir[d];

      kdop._min_values[i] = min(kdop._min_values[i], proj_i);
      kdop._max_values[i] = max(kdop._max_values[i], proj_i);
    }
  }
};

template <typename KDOP, typename Box>
struct expand<KDOPTag, BoxTag, KDOP, Box>
{
  KOKKOS_FUNCTION static void apply(KDOP &kdop, Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<KDOP>;
    static_assert(DIM == 2 || DIM == 3);

    // NOTE if any of the ranges is invalid, the code below would actually
    // expand the KDOP which is not what we want.
    // We may revisit this later and decide passing a valid box becomes a
    // precondition but this would be a breaking change (going from a wide to a
    // narrow contract).
    for (int d = 0; d < DIM; ++d)
      if (box.minCorner()[d] > box.maxCorner()[d])
        return;

    using Point = std::decay_t<decltype(box.minCorner())>;
    if constexpr (DIM == 3)
    {
      auto const xmin = box.minCorner()[0];
      auto const ymin = box.minCorner()[1];
      auto const zmin = box.minCorner()[2];
      auto const xmax = box.maxCorner()[0];
      auto const ymax = box.maxCorner()[1];
      auto const zmax = box.maxCorner()[2];
      for (auto const &point : {
               Point{xmin, ymin, zmin},
               Point{xmin, ymax, zmin},
               Point{xmin, ymin, zmax},
               Point{xmin, ymax, zmax},
               Point{xmax, ymin, zmin},
               Point{xmax, ymax, zmin},
               Point{xmax, ymin, zmax},
               Point{xmax, ymax, zmax},
           })
      {
        Details::expand(kdop, point);
      }
    }
    else
    {
      auto const xmin = box.minCorner()[0];
      auto const ymin = box.minCorner()[1];
      auto const xmax = box.maxCorner()[0];
      auto const ymax = box.maxCorner()[1];
      for (auto const &point : {
               Point{xmin, ymin},
               Point{xmin, ymax},
               Point{xmax, ymin},
               Point{xmax, ymax},
           })
      {
        Details::expand(kdop, point);
      }
    }
  }
};

template <typename Box, typename KDOP>
struct expand<BoxTag, KDOPTag, Box, KDOP>
{
  KOKKOS_FUNCTION static void apply(Box &box, KDOP const &kdop)
  {
    constexpr int DIM = GeometryTraits::dimension_v<KDOP>;

    // WARNING implicit requirement on KDOP first DIM directions
    Box other;
    for (int d = 0; d < DIM; ++d)
    {
      other.minCorner()[d] = kdop._min_values[d];
      other.maxCorner()[d] = kdop._max_values[d];
    }
    Details::expand(box, other);
  }
};

template <typename Box, typename Segment>
struct expand<BoxTag, SegmentTag, Box, Segment>
{
  KOKKOS_FUNCTION static void apply(Box &box, Segment const &segment)
  {
    Details::expand(box, segment._start);
    Details::expand(box, segment._end);
  }
};

// expand a box to include a tetrahedron
template <typename Box, typename Tetrahedron>
struct expand<BoxTag, TetrahedronTag, Box, Tetrahedron>
{
  KOKKOS_FUNCTION static void apply(Box &box, Tetrahedron const &tet)
  {
    Details::expand(box, tet.a);
    Details::expand(box, tet.b);
    Details::expand(box, tet.c);
    Details::expand(box, tet.d);
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
