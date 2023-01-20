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

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::ExperimentalHyperGeometry
{

template <int DIM, class FloatingPoint = float>
struct Point
{
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Point() noexcept = default;

  KOKKOS_FUNCTION
  constexpr auto &operator[](unsigned int i) { return _coords[i]; }

  KOKKOS_FUNCTION
  constexpr auto const &operator[](unsigned int i) const { return _coords[i]; }

  // Initialization is needed to be able to use Point in constexpr
  // TODO: do we want to actually want to zero initialize it? Seems like
  // unnecessary work.
  FloatingPoint _coords[DIM] = {};
};

// Deduction guides
template <class T>
Point(T x, T y) -> Point<2, T>;

template <class T>
Point(T x, T y, T z) -> Point<3, T>;

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class FloatingPoint>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Point<DIM, FloatingPoint>>
{
  static constexpr int value = DIM;
};
template <int DIM, class FloatingPoint>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Point<DIM, FloatingPoint>>
{
  using type = PointTag;
};

#endif
