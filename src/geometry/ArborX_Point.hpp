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
#ifndef ARBORX_POINT_HPP
#define ARBORX_POINT_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
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
KOKKOS_FUNCTION Point(T...)
    -> Point<sizeof...(T), std::conditional_t<
                               (... || std::is_same_v<std::decay_t<T>, double>),
                               double, float>>;

} // namespace ArborX

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<ArborX::Point<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<ArborX::Point<DIM, Coordinate>>
{
  using type = PointTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<ArborX::Point<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
