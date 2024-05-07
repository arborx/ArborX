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
#ifndef ARBORX_DETAILS_KOKKOS_EXT_STD_MEMORY_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_STD_MEMORY_HPP

#include <Kokkos_Macros.hpp>

#include <utility>

namespace ArborX::Details::KokkosExt
{

template <class T, class... Args>
KOKKOS_INLINE_FUNCTION constexpr T *construct_at(T *p, Args &&...args)
{
  return ::new (static_cast<void *>(p)) T(std::forward<Args>(args)...);
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr void destroy_at(T *p)
{
  p->~T();
}

} // namespace ArborX::Details::KokkosExt

#endif
