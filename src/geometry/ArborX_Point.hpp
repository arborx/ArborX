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

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>

#include <utility>

namespace ArborX
{
class Point
{
private:
  struct Data
  {
    float coords[3];
  } _data = {};

  struct Abomination
  {
    double xyz[3];
  };

public:
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Point() noexcept = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Point(Abomination data)
      : Point(static_cast<float>(data.xyz[0]), static_cast<float>(data.xyz[1]),
              static_cast<float>(data.xyz[2]))
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr Point(float x, float y, float z)
      : _data{{x, y, z}}
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr float &operator[](unsigned int i) { return _data.coords[i]; }

  KOKKOS_INLINE_FUNCTION
  constexpr float const &operator[](unsigned int i) const
  {
    return _data.coords[i];
  }
};

template <>
struct GeometryTraits::dimension<ArborX::Point>
{
  static constexpr int value = 3;
};
template <>
struct GeometryTraits::tag<ArborX::Point>
{
  using type = PointTag;
};

} // namespace ArborX

#endif
