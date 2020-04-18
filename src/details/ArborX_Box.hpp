/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_BOX_HPP
#define ARBORX_BOX_HPP

#include <ArborX_DetailsKokkosExt.hpp> // ArithmeticTraits
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
struct Box
{
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Box() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Box(Point const &min_corner, Point const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point &minCorner() { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &minCorner() const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile &minCorner() volatile { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile const &minCorner() volatile const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point &maxCorner() { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &maxCorner() const { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile &maxCorner() volatile { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile const &maxCorner() volatile const { return _max_corner; }

  Point _min_corner = {{KokkosExt::ArithmeticTraits::max<float>::value,
                        KokkosExt::ArithmeticTraits::max<float>::value,
                        KokkosExt::ArithmeticTraits::max<float>::value}};
  Point _max_corner = {{-KokkosExt::ArithmeticTraits::max<float>::value,
                        -KokkosExt::ArithmeticTraits::max<float>::value,
                        -KokkosExt::ArithmeticTraits::max<float>::value}};
};
} // namespace ArborX

#endif
