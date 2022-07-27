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
#ifndef ARBORX_Sphere_HPP
#define ARBORX_Sphere_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{

struct Sphere
{
  KOKKOS_DEFAULTED_FUNCTION
  Sphere() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Sphere(Point const &centroid, double radius) // FIXME
      : _centroid(centroid)
      , _radius(static_cast<float>(radius))
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr Point &centroid() { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &centroid() const { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  constexpr float radius() const { return _radius; }

  Point _centroid = {};
  float _radius = 0.;
};

namespace Experimental
{

template <>
struct GeometryTraits<ArborX::Sphere>
{
  using tag = SphereTag;
  using coordinate_type = float;
  static int const dimension = 3;
};

} // namespace Experimental

} // namespace ArborX

#endif
