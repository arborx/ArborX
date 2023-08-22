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

#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>

#define RBF_DECL(name)                                                         \
  template <int K>                                                             \
  struct __##name;                                                             \
                                                                               \
  template <int K>                                                             \
  static constexpr __##name<K> name                                            \
  {}

#define RBF_DEF(name, n, func)                                                 \
  template <>                                                                  \
  struct __##name<n>                                                           \
  {                                                                            \
    template <typename T>                                                      \
    KOKKOS_INLINE_FUNCTION static T apply(T x)                                 \
    {                                                                          \
      return func;                                                             \
    }                                                                          \
  }

namespace Details
{

RBF_DECL(wendland);
RBF_DEF(wendland, 0, (1 - x) * (1 - x));
RBF_DEF(wendland, 2, (1 - x) * (1 - x) * (1 - x) * (1 - x) * (4 * x + 1));
RBF_DEF(wendland, 4,
        (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
            (35 * x * x + 18 * x + 3));
RBF_DEF(wendland, 6,
        (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
            (1 - x) * (32 * x * x * x + 25 * x * x + 8 * x + 1));

RBF_DECL(wu);
RBF_DEF(wu, 2,
        (1 - x) * (1 - x) * (1 - x) * (1 - x) *
            (3 * x * x * x + 12 * x + 16 * x + 4));
RBF_DEF(wu, 4,
        (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) * (1 - x) *
            (5 * x * x * x * x * x + 30 * x * x * x * x + 72 * x * x * x +
             82 * x * x + 36 * x + 6));

RBF_DECL(buhmann);
RBF_DEF(buhmann, 2,
        2 * x * x * x * x * log(x) - T(7) / 2 * x * x * x * x +
            T(16) / 3 * x * x * x - 2 * x * x + T(1) / 6);
RBF_DEF(buhmann, 3,
        1 * x * x * x * x * x * x * x * x - T(84) / 5 * x * x * x * x * x * x +
            T(1024) / 5 * x * x * x * x * sqrt(x) - 378 * x * x * x * x +
            T(1024) / 5 * x * x * x * sqrt(x) - T(84) / 5 * x * x + 1);
RBF_DEF(buhmann, 4,
        T(99) / 35 * x * x * x * x * x * x * x * x -
            132 * x * x * x * x * x * x +
            T(9216) / 35 * x * x * x * x * x * sqrt(x) -
            T(11264) / 35 * x * x * x * x * sqrt(x) + 198 * x * x * x * x -
            T(396) / 5 * x * x + 1);

} // namespace Details

#undef RBF_DECL
#undef RBF_DEF