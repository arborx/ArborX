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
#ifndef ARBORX_SPHERE_HPP
#define ARBORX_SPHERE_HPP

#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{

template <int DIM = 3>
struct SphereD
{
  static int const dim = DIM;

  KOKKOS_DEFAULTED_FUNCTION
  SphereD() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr SphereD(PointD<DIM> const &centroid, double radius) // FIXME
      : _centroid(centroid)
      , _radius(static_cast<float>(radius))
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> &centroid() { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> const &centroid() const { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  constexpr float radius() const { return _radius; }

  PointD<DIM> _centroid = {};
  float _radius = 0.;
};

using Sphere = SphereD<3>;

} // namespace ArborX

#endif
