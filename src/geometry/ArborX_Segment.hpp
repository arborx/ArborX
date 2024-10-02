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
#ifndef ARBORX_SEGMENT_HPP
#define ARBORX_SEGMENT_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Vector.hpp>
#include <details/ArborX_Algorithms.hpp>

#include <Kokkos_Clamp.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX::Experimental
{
template <int DIM, typename Coordinate = float>
struct Segment
{
  ArborX::Point<DIM, Coordinate> _start;
  ArborX::Point<DIM, Coordinate> _end;
};

template <int DIM, typename Coordinate>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
    Segment(Point<DIM, Coordinate>, Point<DIM, Coordinate>)
        -> Segment<DIM, Coordinate>;

template <int N, typename T>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
Segment(T const (&)[N], T const (&)[N]) -> Segment<N, T>;

} // namespace ArborX::Experimental

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  using type = SegmentTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::Segment<DIM, Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{

using GeometryTraits::BoxTag;
using GeometryTraits::PointTag;
using GeometryTraits::SegmentTag;

template <typename Box, typename Segment>
struct expand<BoxTag, SegmentTag, Box, Segment>
{
  KOKKOS_FUNCTION static void apply(Box &box, Segment const &segment)
  {
    Details::expand(box, segment._start);
    Details::expand(box, segment._end);
  }
};

template <typename Point, typename Segment>
struct distance<PointTag, SegmentTag, Point, Segment>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Segment const &segment)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;

    if (Details::equals(segment._start, segment._end))
      return Details::distance(point, segment._start);

    auto const dir = segment._end - segment._start;

    // The line of the segment [a,b] is parametrized as a + t * (b - a).
    // Find the projection of the point to that line, and clamp it.
    auto t =
        Kokkos::clamp(dir.dot(point - segment._start) / dir.dot(dir),
                      static_cast<Coordinate>(0), static_cast<Coordinate>(1));

    Point projection;
    for (int d = 0; d < DIM; ++d)
      projection[d] = segment._start[d] + t * dir[d];

    return Details::distance(point, projection);
  }
};

template <typename Segment>
struct centroid<SegmentTag, Segment>
{
  KOKKOS_FUNCTION static auto apply(Segment const &segment)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Segment>;
    using Coordinate = GeometryTraits::coordinate_type_t<Segment>;

    // WARNING implicit requirement on KDOP first DIM directions
    Point<DIM, Coordinate> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = (segment._start[d] + segment._end[d]) / 2;
    return point;
  }
};

} // namespace ArborX::Details::Dispatch

#endif
