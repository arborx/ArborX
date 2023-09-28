/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INTERP_DETAILS_COMPACT_RADIAL_BASIS_FUNCTION_HPP
#define ARBORX_INTERP_DETAILS_COMPACT_RADIAL_BASIS_FUNCTION_HPP

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation::CRBF
{

#define CRBF_DECL(NAME)                                                        \
  template <std::size_t>                                                       \
  struct NAME;

#define CRBF_DEF(NAME, N, FUNC)                                                \
  template <>                                                                  \
  struct NAME<N>                                                               \
  {                                                                            \
    template <typename T>                                                      \
    KOKKOS_INLINE_FUNCTION static constexpr T evaluate(T const y)              \
    {                                                                          \
      T const x = Kokkos::min(Kokkos::abs(y), T(1));                           \
      return Kokkos::abs(FUNC);                                                \
    }                                                                          \
  };

CRBF_DECL(Wendland)
CRBF_DEF(Wendland, 0, (1 - x) * (1 - x))
CRBF_DEF(Wendland, 2, (1 - x) * (1 - x) * (1 - x) * (1 - x) * (4 * x + 1))
CRBF_DEF(Wendland, 4,
         (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
             (35 * x * x + 18 * x + 3))
CRBF_DEF(Wendland, 6,
         (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
             (1 - x) * (32 * x * x * x + 25 * x * x + 8 * x + 1))

CRBF_DECL(Wu)
CRBF_DEF(Wu, 2,
         (1 - x) * (1 - x) * (1 - x) * (1 - x) *
             (3 * x * x * x + 12 * x + 16 * x + 4))
CRBF_DEF(Wu, 4,
         (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
             (5 * x * x * x * x * x + 30 * x * x * x * x + 72 * x * x * x +
              82 * x * x + 36 * x + 6))

CRBF_DECL(Buhmann)
CRBF_DEF(Buhmann, 2,
         (x == T(0))
             ? T(1) / 6
             : 2 * x * x * x * x * Kokkos::log(x) - 7 * x * x * x * x / 2 +
                   16 * x * x * x / 3 - 2 * x * x + T(1) / 6)
CRBF_DEF(Buhmann, 3,
         1 * x * x * x * x * x * x * x * x - 84 * x * x * x * x * x * x / 5 +
             1024 * x * x * x * x * Kokkos::sqrt(x) / 5 - 378 * x * x * x * x +
             1024 * x * x * x * Kokkos::sqrt(x) / 5 - 84 * x * x / 5 + 1)
CRBF_DEF(Buhmann, 4,
         99 * x * x * x * x * x * x * x * x / 35 - 132 * x * x * x * x * x * x +
             9216 * x * x * x * x * x * Kokkos::sqrt(x) / 35 -
             11264 * x * x * x * x * Kokkos::sqrt(x) / 35 +
             198 * x * x * x * x - 396 * x * x / 35 + 1)

#undef CRBF_DEF
#undef CRBF_DECL

} // namespace ArborX::Interpolation::CRBF

#endif