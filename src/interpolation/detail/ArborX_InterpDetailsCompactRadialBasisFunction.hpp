/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INTERP_COMPACT_RADIAL_BASIS_FUNCTION_HPP
#define ARBORX_INTERP_COMPACT_RADIAL_BASIS_FUNCTION_HPP

#include <ArborX_GeometryTraits.hpp>
#include <algorithms/ArborX_Distance.hpp>

#include <Kokkos_Core.hpp>

#include <initializer_list>

namespace ArborX::Interpolation
{

namespace Details
{

// Polynomials are represented with the highest coefficient first. For example,
// 3x^2 - 2x + 1 would be {3, -2, 1}
template <typename T>
KOKKOS_INLINE_FUNCTION T
evaluatePolynomial(T const x, std::initializer_list<T> const coeffs)
{
  T eval = 0;
  for (auto const coeff : coeffs)
    eval = x * eval + coeff;
  return eval;
}

} // namespace Details

namespace CRBF
{

#define CRBF_DECL(NAME)                                                        \
  template <std::size_t N>                                                     \
  struct NAME;

#define CRBF_DEF(NAME, N, FUNC)                                                \
  template <>                                                                  \
  struct NAME<N>                                                               \
  {                                                                            \
    template <std::size_t DIM, typename T>                                     \
    KOKKOS_INLINE_FUNCTION static constexpr T evaluate(T const y)              \
    {                                                                          \
      /* We force the input to be between 0 and 1.                             \
         Because CRBF(-a) = CRBF(a) = CRBF(|a|), we take the absolute value    \
         and clamp the range to [0, 1] before entering in the definition of    \
         the CRBF.                                                             \
         We also template the internal function on the dimension as CRBFs      \
         depend on the point's dimensionality. */                              \
      T const x = Kokkos::min(Kokkos::abs(y), T(1));                           \
      return Kokkos::abs(FUNC);                                                \
    }                                                                          \
  };

#define CRBF_POLY(...) Details::evaluatePolynomial<T>(x, {__VA_ARGS__})
#define CRBF_POW(X, N) Kokkos::pow(X, N)

CRBF_DECL(Wendland)
CRBF_DEF(Wendland, 0, CRBF_POW(CRBF_POLY(-1, 1), 2))
CRBF_DEF(Wendland, 2, CRBF_POW(CRBF_POLY(-1, 1), 4) * CRBF_POLY(4, 1))
CRBF_DEF(Wendland, 4, CRBF_POW(CRBF_POLY(-1, 1), 6) * CRBF_POLY(35, 18, 3))
CRBF_DEF(Wendland, 6, CRBF_POW(CRBF_POLY(-1, 1), 8) * CRBF_POLY(32, 25, 8, 1))

CRBF_DECL(Wu)
CRBF_DEF(Wu, 2, CRBF_POW(CRBF_POLY(-1, 1), 4) * CRBF_POLY(3, 12, 16, 4))
CRBF_DEF(Wu, 4, CRBF_POW(CRBF_POLY(-1, 1), 6) * CRBF_POLY(5, 30, 72, 82, 36, 6))

CRBF_DECL(Buhmann)
CRBF_DEF(Buhmann, 2,
         (x == T(0)) ? T(1) / 6
                     : CRBF_POLY(12 * Kokkos::log(x) - 21, 32, -12, 0, 1) / 6)
CRBF_DEF(Buhmann, 3,
         CRBF_POLY(5, 0, -84, 0, 1024 * Kokkos::sqrt(x) - 1890,
                   1024 * Kokkos::sqrt(x), -84, 0, 5) /
             5)
CRBF_DEF(Buhmann, 4,
         CRBF_POLY(99, 0, -4620, 9216 * Kokkos::sqrt(x),
                   -11264 * Kokkos::sqrt(x) + 6930, 0, -396, 0, 35) /
             35)

#undef CRBF_POW
#undef CRBF_POLY
#undef CRBF_DEF
#undef CRBF_DECL

template <typename CRBFunc, typename Point>
KOKKOS_INLINE_FUNCTION constexpr auto evaluate(Point const &point)
{
  static_assert(GeometryTraits::is_point_v<Point>, "Point must be a point");
  constexpr std::size_t dim = GeometryTraits::dimension_v<Point>;
  return CRBFunc::template evaluate<dim>(distance(point, Point{}));
}

} // namespace CRBF

} // namespace ArborX::Interpolation

#endif
