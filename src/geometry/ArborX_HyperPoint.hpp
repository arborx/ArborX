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

#ifndef ARBORX_HYPERPOINT_HPP
#define ARBORX_HYPERPOINT_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp> // isfinite, sqrt

namespace ArborX::ExperimentalHyperGeometry
{

template <int DIM, class Coordinate = float>
struct Point
{
  static_assert(DIM > 0);

  KOKKOS_FUNCTION
  constexpr auto &operator[](unsigned int i) { return _coords[i]; }

  KOKKOS_FUNCTION
  constexpr auto const &operator[](unsigned int i) const { return _coords[i]; }

  // Initialization is needed to be able to use Point in constexpr
  // TODO: do we want to actually want to zero initialize it? Seems like
  // unnecessary work.
  Coordinate _coords[DIM] = {};
};

template <typename... T>
Point(T...)
    -> Point<sizeof...(T), std::conditional_t<
                               (... || std::is_same_v<std::decay_t<T>, double>),
                               double, float>>;

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  using type = PointTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Point<DIM, Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{
using GeometryTraits::PointTag;

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

// distance point-point
template <typename Point1, typename Point2>
struct distance<PointTag, PointTag, Point1, Point2>
{
  KOKKOS_FUNCTION static auto apply(Point1 const &a, Point2 const &b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point1>;
    // Points may have different coordinate types. Try using implicit
    // conversion to get the best one.
    using Coordinate = decltype(b[0] - a[0]);
    Coordinate distance_squared = 0;
    for (int d = 0; d < DIM; ++d)
    {
      auto tmp = b[d] - a[d];
      distance_squared += tmp * tmp;
    }
    return Kokkos::sqrt(distance_squared);
  }
};

template <typename Point>
struct centroid<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr auto apply(Point const &point)
  {
    return point;
  }
};

} // namespace ArborX::Details::Dispatch

#endif
