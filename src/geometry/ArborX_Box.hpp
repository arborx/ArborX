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

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_ReductionIdentity.hpp>

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
  constexpr Box(Point<3> const &min_corner, Point<3> const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr auto &minCorner() { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr auto const &minCorner() const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr auto &maxCorner() { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr auto const &maxCorner() const { return _max_corner; }

  Point<3> _min_corner = {
      {Details::KokkosExt::ArithmeticTraits::finite_max<float>::value,
       Details::KokkosExt::ArithmeticTraits::finite_max<float>::value,
       Details::KokkosExt::ArithmeticTraits::finite_max<float>::value}};
  Point<3> _max_corner = {
      {Details::KokkosExt::ArithmeticTraits::finite_min<float>::value,
       Details::KokkosExt::ArithmeticTraits::finite_min<float>::value,
       Details::KokkosExt::ArithmeticTraits::finite_min<float>::value}};

// FIXME Temporary workaround until we clarify requirements on the Kokkos side.
#if defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_SYCL)
private:
  [[deprecated]] friend KOKKOS_FUNCTION Box operator+(Box box, Box const &other)
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
template <>
struct ArborX::GeometryTraits::coordinate_type<ArborX::Box>
{
  using type = float;
};

} // namespace ArborX

template <>
struct [[deprecated]] Kokkos::reduction_identity<ArborX::Box>
{
  KOKKOS_FUNCTION static ArborX::Box sum() { return {}; }
};

#endif
