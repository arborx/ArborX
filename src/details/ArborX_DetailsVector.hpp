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
#ifndef ARBORX_DETAILS_VECTOR_HPP
#define ARBORX_DETAILS_VECTOR_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

namespace ArborX::Details
{

// Same as Point, but a different type (can't inherit from Point as it would
// run into constexpr and inability to aggregate initialize)
template <int DIM, typename Coordinate = float>
struct Vector
{
  static_assert(DIM > 0);
  // Initialization is needed to be able to use Vector in constexpr
  // TODO: do we want to actually want to zero initialize it? Seems like
  // unnecessary work.
  Coordinate _coords[DIM] = {};

  KOKKOS_FUNCTION
  constexpr auto &operator[](unsigned int i) { return _coords[i]; }

  KOKKOS_FUNCTION
  constexpr auto const &operator[](unsigned int i) const { return _coords[i]; }

  KOKKOS_FUNCTION
  constexpr auto dot(Vector const &w) const
  {
    auto const &v = *this;

    Coordinate r = 0;
    for (int d = 0; d < DIM; ++d)
      r += v[d] * w[d];
    return r;
  }

  KOKKOS_FUNCTION
  auto norm() const { return Kokkos::sqrt(dot(*this)); }

  KOKKOS_FUNCTION
  constexpr auto cross(Vector const &w) const
  {
    static_assert(DIM == 3);

    auto const &v = *this;
    return Vector{v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2],
                  v[0] * w[1] - v[1] * w[0]};
  }

  friend KOKKOS_FUNCTION constexpr bool operator==(Vector const &v,
                                                   Vector const &w)
  {
    bool match = true;
    for (int d = 0; d < DIM; ++d)
      match &= (v[d] == w[d]);
    return match;
  }
};

template <typename T, typename... U>
Vector(T, U...) -> Vector<1 + sizeof...(U), T>;

} // namespace ArborX::Details

template <typename Point, typename Enable = std::enable_if_t<
                              ArborX::GeometryTraits::is_point_v<Point>>>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(Point const &end,
                                                Point const &begin)
{
  namespace GT = ArborX::GeometryTraits;
  constexpr int DIM = GT::dimension_v<Point>;
  ArborX::Details::Vector<DIM, GT::coordinate_type_t<Point>> v;
  for (int d = 0; d < DIM; ++d)
    v[d] = end[d] - begin[d];
  return v;
}

#endif
