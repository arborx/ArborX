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

#ifndef ARBORX_BOX_HPP
#define ARBORX_BOX_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>
#if __has_include(<Kokkos_ReductionIdentity.hpp>) // FIXME requires Kokkos 4.0
#include <Kokkos_ReductionIdentity.hpp>
#endif

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
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr Point &minCorner() { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &minCorner() const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point &maxCorner() { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &maxCorner() const { return _max_corner; }

  Point _min_corner = {{KokkosExt::ArithmeticTraits::finite_max<float>::value,
                        KokkosExt::ArithmeticTraits::finite_max<float>::value,
                        KokkosExt::ArithmeticTraits::finite_max<float>::value}};
  Point _max_corner = {{KokkosExt::ArithmeticTraits::finite_min<float>::value,
                        KokkosExt::ArithmeticTraits::finite_min<float>::value,
                        KokkosExt::ArithmeticTraits::finite_min<float>::value}};

  KOKKOS_FUNCTION Box &operator+=(Box const &other)
  {
    ArborX::Details::expand(*this, other);
    return *this;
  }

  KOKKOS_FUNCTION Box &operator+=(Point const &point)
  {
    ArborX::Details::expand(*this, point);
    return *this;
  }

// FIXME Temporary workaround until we clarify requirements on the Kokkos side.
#if defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_SYCL)
private:
  friend KOKKOS_FUNCTION Box operator+(Box box, Box const &other)
  {
    return box += other;
  }
#endif
};

template <>
struct GeometryTraits::dimension<ArborX::Box>
{
  static constexpr int value = 3;
};
template <>
struct GeometryTraits::tag<ArborX::Box>
{
  using type = BoxTag;
};

} // namespace ArborX

template <>
struct Kokkos::reduction_identity<ArborX::Box>
{
  KOKKOS_FUNCTION static ArborX::Box sum() { return {}; }
};

#endif
