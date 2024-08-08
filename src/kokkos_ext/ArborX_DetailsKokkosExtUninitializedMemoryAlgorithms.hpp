/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_KOKKOS_EXT_UNINITIALIZED_MEMORY_ALGORITHMS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_UNINITIALIZED_MEMORY_ALGORITHMS_HPP

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX::Details::KokkosExt
{

template <class T, class... Args>
KOKKOS_FUNCTION constexpr T *construct_at(T *p, Args &&...args)
{
  return ::new (const_cast<void *>(static_cast<void const volatile *>(p)))
      T((Args &&)args...);
}

template <class T>
KOKKOS_FUNCTION constexpr void destroy_at(T *p)
{
  KOKKOS_ASSERT(p != nullptr);
  p->~T();
}

} // namespace ArborX::Details::KokkosExt

#endif
