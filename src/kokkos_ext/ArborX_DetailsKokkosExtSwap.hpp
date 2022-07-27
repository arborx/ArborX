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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_SWAP_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_SWAP_HPP

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility>

namespace KokkosExt
{

template <class T>
KOKKOS_FUNCTION constexpr void
swap(T &a, T &b) noexcept(std::is_nothrow_move_constructible<T>::value
                              &&std::is_nothrow_move_assignable<T>::value)
{
  T tmp = std::move(a);
  a = std::move(b);
  b = std::move(tmp);
}

} // namespace KokkosExt

#endif
