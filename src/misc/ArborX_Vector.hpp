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
#include <ArborX_Point.hpp>

#include <Kokkos_Assert.hpp>
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

  template <typename Precision = Coordinate>
  KOKKOS_FUNCTION constexpr auto dot(Vector const &w) const
  {
    auto const &v = *this;

    Precision r = 0;
    for (int d = 0; d < DIM; ++d)
      r += static_cast<Precision>(v[d]) * static_cast<Precision>(w[d]);
    return r;
  }

  template <typename Precision = Coordinate>
  KOKKOS_FUNCTION auto norm() const
  {
    return Kokkos::sqrt(dot<Precision>(*this));
  }

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

template <typename Precision, int DIM, typename Coordinate>
KOKKOS_INLINE_FUNCTION auto normalize(Vector<DIM, Coordinate> const &v)
{
  auto const magv = v.template norm<Precision>();
  KOKKOS_ASSERT(magv > 0);

  Vector<DIM, Coordinate> w;
  for (int d = 0; d < DIM; ++d)
    w[d] = v[d] / magv;
  return w;
}

template <int DIM, typename Coordinate>
KOKKOS_INLINE_FUNCTION auto normalize(Vector<DIM, Coordinate> const &v)
{
  return normalize<Coordinate>(v);
}

template <typename T, typename... U>
Vector(T, U...) -> Vector<1 + sizeof...(U), T>;

} // namespace ArborX::Details

// FIXME: remove second template argument when ExperimentalHyperGeometry switch
// happens
template <typename Point, typename Point2,
          typename Enable =
              std::enable_if_t<ArborX::GeometryTraits::is_point_v<Point> &&
                               ArborX::GeometryTraits::is_point_v<Point2>>>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(Point const &end,
                                                Point2 const &begin)
{
  namespace GT = ArborX::GeometryTraits;
  constexpr int DIM = GT::dimension_v<Point>;
  static_assert(GT::dimension_v<Point2> == DIM);
  ArborX::Details::Vector<DIM, GT::coordinate_type_t<Point>> v;
  for (int d = 0; d < DIM; ++d)
    v[d] = end[d] - begin[d];
  return v;
}

#endif
