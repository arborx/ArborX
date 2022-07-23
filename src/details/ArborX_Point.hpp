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

#ifndef ARBORX_DETAILS_POINT_HPP
#define ARBORX_DETAILS_POINT_HPP

#include <Kokkos_Macros.hpp>

#include <utility>

namespace ArborX
{

template <int DIM = 3>
class PointD
{
private:
  struct Data
  {
    float coords[DIM];
  } _data = {};

  struct Abomination
  {
    double xyz[DIM];
  };

public:
  static int const dim = DIM;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr PointD() noexcept = default;

  KOKKOS_INLINE_FUNCTION constexpr PointD(Abomination data)
  {
    for (int d = 0; d < DIM; ++d)
      _data.coords[d] = static_cast<float>(data.xyz[d]);
  }

  template <int D = DIM, std::enable_if_t<D == 2> = 0>
  KOKKOS_INLINE_FUNCTION constexpr PointD(float x, float y)
      : _data{{x, y}}
  {}

  template <int D = DIM, std::enable_if_t<D == 3, int> = 0>
  KOKKOS_INLINE_FUNCTION constexpr PointD(float x, float y, float z)
      : _data{{x, y, z}}
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr float &operator[](unsigned int i) { return _data.coords[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr float const &operator[](unsigned int i) const
  {
    return _data.coords[i];
  }

  KOKKOS_INLINE_FUNCTION
  float volatile &operator[](unsigned int i) volatile
  {
    return _data.coords[i];
  }

  KOKKOS_INLINE_FUNCTION
  float const volatile &operator[](unsigned int i) const volatile
  {
    return _data.coords[i];
  }
};

using Point = PointD<3>;

} // namespace ArborX

#endif
